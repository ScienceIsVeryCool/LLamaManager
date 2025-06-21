#!/usr/bin/env python3
"""
Scenario-based Ollama Agent Execution Engine
Processes JSON scenario files to orchestrate multi-agent interactions
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScenarioError(Exception):
    """Base exception for scenario execution errors"""
    pass


class TemplateError(ScenarioError):
    """Raised when template substitution fails"""
    pass


class ActionError(ScenarioError):
    """Raised when action execution fails"""
    pass


@dataclass
class AgentInstance:
    """Runtime agent instance"""
    name: str
    template_name: str
    model: str
    temperature: float
    system_prompt: str
    context_mode: str = "clean"  # clean, append, rolling
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    _client: Optional[ollama.Client] = None
    
    @property
    def client(self) -> ollama.Client:
        if self._client is None:
            self._client = ollama.Client()
        return self._client
    
    async def query(self, prompt: str, timeout: int = 300) -> str:
        """Send query to model with timeout"""
        messages = []
        
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        if self.context_mode == "append":
            messages.extend(self.conversation_history)
        elif self.context_mode == "rolling" and self.conversation_history:
            # Keep last 10 messages for rolling context
            messages.extend(self.conversation_history[-10:])
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Add timeout to the chat call
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.chat,
                    model=self.model,
                    messages=messages,
                    options={"temperature": self.temperature}
                ),
                timeout=timeout
            )
            
            result = response['message']['content']
            
            # Update history
            self.conversation_history.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": result}
            ])
            
            return result
            
        except asyncio.TimeoutError:
            raise ActionError(f"Query timed out after {timeout}s for agent {self.name}")
        except Exception as e:
            raise ActionError(f"Query failed for agent {self.name}: {e}")
    
    def clear_context(self):
        """Clear conversation history"""
        self.conversation_history.clear()
    
    def get_last_output(self) -> Optional[str]:
        """Get last assistant response"""
        for msg in reversed(self.conversation_history):
            if msg["role"] == "assistant":
                return msg["content"]
        return None


class TemplateEngine:
    """Handles template variable substitution"""
    
    @staticmethod
    def substitute(template: str, context: Dict[str, Any]) -> str:
        """Substitute template variables with values from context"""
        def replace_var(match):
            var_path = match.group(1)
            
            # Handle function calls like lastOutput('agent')
            if '(' in var_path:
                return TemplateEngine._evaluate_function(var_path, context)
            
            # Handle dot notation like outputs.step_id
            parts = var_path.split('.')
            value = context
            
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    raise TemplateError(f"Variable not found: {var_path}")
            
            return str(value)
        
        # Find all {{variable}} patterns
        pattern = r'\{\{([^}]+)\}\}'
        return re.sub(pattern, replace_var, template)
    
    @staticmethod
    def _evaluate_function(func_call: str, context: Dict[str, Any]) -> str:
        """Evaluate template functions"""
        match = re.match(r"(\w+)\('([^']+)'\)", func_call)
        if not match:
            raise TemplateError(f"Invalid function syntax: {func_call}")
        
        func_name, arg = match.groups()
        
        if func_name == "lastOutput":
            agents = context.get('agents', {})
            if arg in agents:
                output = agents[arg].get_last_output()
                if output:
                    return output
            raise TemplateError(f"No output found for agent: {arg}")
        
        raise TemplateError(f"Unknown function: {func_name}")


class ScenarioExecutor:
    """Main scenario execution engine"""
    
    def __init__(self, scenario_path: str):
        self.scenario_path = scenario_path
        self.scenario = self._load_scenario()
        self.agents: Dict[str, AgentInstance] = {}
        self.outputs: Dict[str, str] = {}
        self.current_model: Optional[str] = None
        
    def _load_scenario(self) -> Dict[str, Any]:
        """Load and validate scenario file"""
        try:
            with open(self.scenario_path, 'r') as f:
                scenario = json.load(f)
            
            # Basic validation
            required = ['agentTemplates', 'actionTemplates', 'execution']
            for key in required:
                if key not in scenario:
                    raise ScenarioError(f"Missing required section: {key}")
            
            return scenario
            
        except json.JSONDecodeError as e:
            raise ScenarioError(f"Invalid JSON in scenario file: {e}")
        except FileNotFoundError:
            raise ScenarioError(f"Scenario file not found: {self.scenario_path}")
    
    def _ensure_model_loaded(self, model: str):
        """Ensure only one model is loaded at a time"""
        if self.current_model and self.current_model != model:
            logger.info(f"Unloading model: {self.current_model}")
            self._unload_model(self.current_model)
        
        if self.current_model != model:
            logger.info(f"Loading model: {model}")
            self.current_model = model
    
    def _unload_model(self, model: str):
        """Unload a model from memory"""
        try:
            import requests
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=30
            )
        except Exception as e:
            logger.warning(f"Failed to unload model {model}: {e}")
    
    async def execute(self):
        """Execute the scenario"""
        config = self.scenario.get('config', {})
        
        # Setup logging
        log_level = getattr(logging, config.get('logLevel', 'INFO').upper())
        logging.getLogger().setLevel(log_level)
        
        # Setup output directory
        output_dir = Path(config.get('outputDirectory', './results'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Execute steps
        execution_steps = self.scenario['execution']
        
        logger.info(f"Starting scenario: {self.scenario.get('scenario', {}).get('name', 'Unnamed')}")
        
        try:
            await self._execute_steps(execution_steps)
        finally:
            # Cleanup
            if self.current_model:
                self._unload_model(self.current_model)
    
    async def _execute_steps(self, steps: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None):
        """Execute a list of steps"""
        if context is None:
            context = {}
        
        for step in steps:
            await self._execute_step(step, context)
    
    async def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]):
        """Execute a single step"""
        step_id = step.get('id', 'unnamed')
        action = step['action']
        
        logger.info(f"Executing step: {step_id} (action: {action})")
        
        # Build context for template substitution
        template_context = {
            'outputs': self.outputs,
            'agents': self.agents,
            **context
        }
        
        # Execute based on action type
        if action == 'createAgent':
            await self._create_agent(step, template_context)
        elif action == 'loop':
            await self._execute_loop(step, template_context)
        elif action == 'saveToFile':
            await self._save_to_file(step, template_context)
        elif action in self.scenario['actionTemplates']:
            await self._execute_action_template(step, template_context)
        else:
            raise ActionError(f"Unknown action: {action}")
    
    async def _create_agent(self, step: Dict[str, Any], context: Dict[str, Any]):
        """Create an agent instance"""
        params = step['params']
        template_name = params['template']
        instance_name = params['instanceName']
        
        if template_name not in self.scenario['agentTemplates']:
            raise ActionError(f"Unknown agent template: {template_name}")
        
        template = self.scenario['agentTemplates'][template_name]
        
        agent = AgentInstance(
            name=instance_name,
            template_name=template_name,
            model=template['model'],
            temperature=template.get('temperature', 0.7),
            system_prompt=template.get('systemPrompt', ''),
            context_mode=template.get('defaultContext', 'clean')
        )
        
        self.agents[instance_name] = agent
        logger.info(f"Created agent: {instance_name} (model: {agent.model})")
    
    async def _execute_action_template(self, step: Dict[str, Any], context: Dict[str, Any]):
        """Execute an action template"""
        action_name = step['action']
        agent_name = step.get('agent')
        params = step.get('params', {})
        
        if agent_name not in self.agents:
            raise ActionError(f"Unknown agent: {agent_name}")
        
        agent = self.agents[agent_name]
        action_template = self.scenario['actionTemplates'][action_name]
        
        # Ensure model is loaded
        self._ensure_model_loaded(agent.model)
        
        # Substitute template variables in parameters
        substituted_params = {}
        for key, value in params.items():
            if isinstance(value, str):
                substituted_params[key] = TemplateEngine.substitute(value, context)
            else:
                substituted_params[key] = value
        
        # Build prompt from template
        prompt_template = action_template['promptTemplate']
        prompt = TemplateEngine.substitute(prompt_template, substituted_params)
        
        # Execute query
        logger.info(f"Agent {agent_name} executing: {action_name}")
        result = await agent.query(prompt)
        
        # Store output
        if 'output' in step:
            self.outputs[step['output']] = result
            logger.debug(f"Stored output '{step['output']}': {len(result)} chars")
        
        # Save intermediate outputs if configured
        if self.scenario.get('config', {}).get('saveIntermediateOutputs', False):
            output_dir = Path(self.scenario['config']['outputDirectory'])
            output_file = output_dir / f"{step.get('id', action_name)}_{agent_name}.md"
            output_file.write_text(result, encoding='utf-8')
    
    async def _execute_loop(self, step: Dict[str, Any], context: Dict[str, Any]):
        """Execute a loop"""
        iterations = step['iterations']
        loop_steps = step['steps']
        
        for i in range(iterations):
            logger.info(f"Loop iteration {i + 1}/{iterations}")
            
            # Create context with iteration variable
            loop_context = {
                **context,
                'iteration': i + 1
            }
            
            # Process steps with iteration substitution
            processed_steps = []
            for loop_step in loop_steps:
                processed_step = {}
                for key, value in loop_step.items():
                    if isinstance(value, str):
                        # Substitute iteration variable
                        value = value.replace('{{iteration}}', str(i + 1))
                    processed_step[key] = value
                processed_steps.append(processed_step)
            
            await self._execute_steps(processed_steps, loop_context)
    
    async def _save_to_file(self, step: Dict[str, Any], context: Dict[str, Any]):
        """Save content to file"""
        params = step['params']
        
        # Substitute template variables
        content = TemplateEngine.substitute(params['content'], context)
        filename = params['filename']
        
        output_dir = Path(self.scenario.get('config', {}).get('outputDirectory', './results'))
        output_path = output_dir / filename
        
        output_path.write_text(content, encoding='utf-8')
        logger.info(f"Saved output to: {output_path}")


async def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python scenario_engine.py <scenario.json>")
        sys.exit(1)
    
    scenario_path = sys.argv[1]
    
    try:
        executor = ScenarioExecutor(scenario_path)
        await executor.execute()
        logger.info("Scenario execution completed successfully")
        
    except ScenarioError as e:
        logger.error(f"Scenario error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
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
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count (approximately 4 chars per token)"""
        return len(text) // 4
    
    def _trim_messages_to_fit(self, messages: List[Dict[str, str]], max_tokens: int = 8000) -> List[Dict[str, str]]:
        """Trim messages from the beginning to fit within token limit"""
        if not messages:
            return messages
            
        # Always keep system prompt if present
        system_messages = [m for m in messages if m['role'] == 'system']
        other_messages = [m for m in messages if m['role'] != 'system']
        
        # Estimate current token usage
        total_tokens = sum(self._estimate_tokens(m['content']) for m in messages)
        
        if total_tokens <= max_tokens:
            return messages
        
        logger.warning(f"Context window approaching limit ({total_tokens} estimated tokens), trimming history for agent {self.name}")
        
        # Keep system messages and trim from the beginning of other messages
        trimmed_messages = system_messages.copy()
        
        # Always keep the last message (the current prompt) if it exists
        current_prompt = None
        if other_messages and other_messages[-1]['role'] == 'user':
            current_prompt = other_messages[-1]
            other_messages = other_messages[:-1]
        
        # Remove oldest messages until we fit
        while other_messages and total_tokens > max_tokens:
            removed = other_messages.pop(0)
            total_tokens -= self._estimate_tokens(removed['content'])
        
        # Add back remaining messages
        trimmed_messages.extend(other_messages)
        if current_prompt:
            trimmed_messages.append(current_prompt)
        
        return trimmed_messages
    
    async def query(self, prompt: str, timeout: int = 300, max_context_tokens: int = 8000) -> str:
        """Send query to model with timeout and context window management"""
        messages = []
        
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        if self.context_mode == "append":
            messages.extend(self.conversation_history)
        elif self.context_mode == "rolling" and self.conversation_history:
            # Keep last 10 messages for rolling context
            messages.extend(self.conversation_history[-10:])
        
        messages.append({"role": "user", "content": prompt})
        
        # Trim messages if needed to fit context window
        messages = self._trim_messages_to_fit(messages, max_context_tokens)
        
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
    def extract_variables(template: str) -> List[str]:
        """Extract all variable names from a template"""
        pattern = r'\{\{([^}]+)\}\}'
        matches = re.findall(pattern, template)
        
        variables = []
        for match in matches:
            # Skip function calls
            if '(' in match:
                continue
            # Skip dot notation (these come from context like outputs.x)
            if '.' in match:
                continue
            variables.append(match)
        
        return list(set(variables))  # Remove duplicates
    
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
            # Try to get the Ollama client's base URL, fallback to localhost
            base_url = "http://localhost:11434"
            try:
                # Check if we can get the base URL from the client
                client = ollama.Client()
                if hasattr(client, '_base_url'):
                    base_url = client._base_url
            except:
                pass  # Use default localhost
                
            import requests
            requests.post(
                f"{base_url}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=30
            )
            logger.debug(f"Successfully unloaded model: {model}")
        except Exception as e:
            logger.warning(f"Failed to unload model {model}: {e}")
    
    def _validate_action_params(self, prompt_template: str, params: Dict[str, Any]) -> None:
        """Validate that all template variables have corresponding parameters"""
        required_vars = TemplateEngine.extract_variables(prompt_template)
        
        missing = [var for var in required_vars if var not in params]
        if missing:
            raise ActionError(f"Missing required parameters: {', '.join(missing)}")
    
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
        
        for i, step in enumerate(steps):
            await self._execute_step(step, context, step_index=i)
    
    async def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any], step_index: int = 0):
        """Execute a single step"""
        step_id = step.get('id', f'step_{step_index}')
        action = step['action']
        
        logger.info(f"Executing: {action} ({step_id})")
        
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
        elif action == 'clearContext':
            await self._clear_context(step, template_context)
        elif action in self.scenario['actionTemplates']:
            await self._execute_action_template(step, template_context, step_id)
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
    
    async def _execute_action_template(self, step: Dict[str, Any], context: Dict[str, Any], step_id: str):
        """Execute an action template"""
        action_name = step['action']
        agent_name = step.get('agent')
        params = step.get('params', {})
        
        if agent_name not in self.agents:
            raise ActionError(f"Unknown agent: {agent_name}")
        
        agent = self.agents[agent_name]
        action_template = self.scenario['actionTemplates'][action_name]
        
        # Validate parameters against template variables
        prompt_template = action_template['promptTemplate']
        self._validate_action_params(prompt_template, params)
        
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
        prompt = TemplateEngine.substitute(prompt_template, substituted_params)
        
        # Get timeout from config (with default)
        config = self.scenario.get('config', {})
        timeout = config.get('queryTimeout', 300)
        max_context = config.get('maxContextTokens', 8000)
        
        # Execute query
        logger.info(f"Agent {agent_name} executing: {action_name}")
        result = await agent.query(prompt, timeout=timeout, max_context_tokens=max_context)
        
        # Store output if specified
        if 'output' in step:
            self.outputs[step['output']] = result
            logger.debug(f"Stored output '{step['output']}': {len(result)} chars")
        
        # Save intermediate outputs if configured
        if self.scenario.get('config', {}).get('saveIntermediateOutputs', False):
            output_dir = Path(self.scenario['config']['outputDirectory'])
            output_file = output_dir / f"{step_id}_{agent_name}.md"
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
                    elif isinstance(value, dict):
                        # Recursively process nested dictionaries (like params)
                        processed_dict = {}
                        for k, v in value.items():
                            if isinstance(v, str):
                                processed_dict[k] = v.replace('{{iteration}}', str(i + 1))
                            else:
                                processed_dict[k] = v
                        value = processed_dict
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
    
    async def _clear_context(self, step: Dict[str, Any], context: Dict[str, Any]):
        """Clear conversation history for specified agent"""
        agent_name = step.get('agent')
        
        if not agent_name:
            raise ActionError("clearContext requires an 'agent' field")
        
        if agent_name not in self.agents:
            raise ActionError(f"Unknown agent: {agent_name}")
        
        self.agents[agent_name].clear_context()
        logger.info(f"Cleared context for agent: {agent_name}")


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
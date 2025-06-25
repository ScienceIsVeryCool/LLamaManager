#!/usr/bin/env python3
"""
Simplified Scenario-based Ollama Agent Execution Engine
Processes JSON scenario files to orchestrate multi-agent interactions
"""

import asyncio
import json
import logging
import re
import subprocess
import os
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


class ValidationError(ScenarioError):
    """Raised when scenario validation fails"""
    pass


class ActionError(ScenarioError):
    """Raised when action execution fails"""
    pass


@dataclass
class AgentInstance:
    """Runtime agent instance"""
    name: str
    model: str
    temperature: float
    personality: str
    max_context_tokens: int = 16000
    query_timeout: int = 300
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
    
    def _trim_messages_to_fit(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Trim messages from the beginning to fit within token limit"""
        if not messages:
            return messages
            
        # Always keep system prompt if present
        system_messages = [m for m in messages if m['role'] == 'system']
        other_messages = [m for m in messages if m['role'] != 'system']
        
        # Estimate current token usage
        total_tokens = sum(self._estimate_tokens(m['content']) for m in messages)
        
        if total_tokens <= self.max_context_tokens:
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
        while other_messages and total_tokens > self.max_context_tokens:
            removed = other_messages.pop(0)
            total_tokens -= self._estimate_tokens(removed['content'])
        
        # Add back remaining messages
        trimmed_messages.extend(other_messages)
        if current_prompt:
            trimmed_messages.append(current_prompt)
        
        return trimmed_messages
    
    async def query(self, prompt: str) -> str:
        """Send query to model with timeout and context window management"""
        messages = []
        
        if self.personality:
            messages.append({"role": "system", "content": self.personality})
        
        # Add all conversation history (will be trimmed if needed)
        if self.conversation_history:
            messages.extend(self.conversation_history)
        
        messages.append({"role": "user", "content": prompt})
        
        # Trim messages if needed to fit context window
        messages = self._trim_messages_to_fit(messages)
        
        try:
            # Add timeout to the chat call using agent-specific timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.chat,
                    model=self.model,
                    messages=messages,
                    options={"temperature": self.temperature}
                ),
                timeout=self.query_timeout
            )
            
            result = response['message']['content']
            
            # Update history - always append to conversation history
            self.conversation_history.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": result}
            ])
            
            return result
            
        except asyncio.TimeoutError:
            raise ActionError(f"Query timed out after {self.query_timeout}s for agent {self.name}")
        except Exception as e:
            raise ActionError(f"Query failed for agent {self.name}: {e}")
    
    def clear_context(self):
        """Clear conversation history"""
        self.conversation_history.clear()


class ScenarioExecutor:
    """Main scenario execution engine"""
    
    def __init__(self, scenario_path: str):
        self.scenario_path = scenario_path
        self.scenario = self._load_scenario()
        # Get working directory from config
        config = self.scenario.get('config', {})
        self.output_dir = Path(config.get('workDir', './results'))
        self.backup_dir = self.output_dir / '.backup'
        self.agents: Dict[str, AgentInstance] = {}
        self.current_model: Optional[str] = None
        self._validate_scenario()
        
    def _load_scenario(self) -> Dict[str, Any]:
        """Load and validate scenario file structure"""
        try:
            with open(self.scenario_path, 'r') as f:
                scenario = json.load(f)
            
            # Basic structure validation
            required = ['agents', 'actions', 'workflow']
            for key in required:
                if key not in scenario:
                    raise ScenarioError(f"Missing required section: {key}")
            
            return scenario
            
        except json.JSONDecodeError as e:
            raise ScenarioError(f"Invalid JSON in scenario file: {e}")
        except FileNotFoundError:
            raise ScenarioError(f"Scenario file not found: {self.scenario_path}")
    
    def _validate_scenario(self):
        """Validate scenario consistency"""
        # Collect all defined agent names and action names
        defined_agents = set(self.scenario['agents'].keys())
        defined_actions = set(self.scenario['actions'].keys())
        
        
        # Check workflow steps
        for i, step in enumerate(self.scenario['workflow']):
            step_id = step.get('id', f'step_{i}')
            
            # Validate agent exists
            if 'agent' in step:
                agent_name = step['agent']
                if agent_name not in defined_agents:
                    raise ValidationError(f"Step '{step_id}': Unknown agent '{agent_name}'. Available agents: {', '.join(sorted(defined_agents))}")
                

            
            # Validate action exists
            action_name = step.get('action')
            if action_name and action_name not in ['createAgent', 'loop', 'clear_context', 'run_python']:
                if action_name not in defined_actions:
                    raise ValidationError(f"Step '{step_id}': Unknown action '{action_name}'. Available actions: {', '.join(sorted(defined_actions))}")
                
                # Validate input count matches placeholders in action prompt
                action_prompt = self.scenario['actions'][action_name]['prompt']
                placeholder_count = len(re.findall(r'\{\{(\d+)\}\}', action_prompt))
                input_count = len(step.get('inputs', []))
                
                if input_count != placeholder_count:
                    raise ValidationError(
                        f"Step '{step_id}': Action '{action_name}' expects {placeholder_count} inputs but got {input_count}"
                    )
        
        # Additional validation for loop steps
        self._validate_loop_steps(self.scenario['workflow'])

    def _validate_loop_steps(self, steps):
        """Recursively validate loop steps"""
        for i, step in enumerate(steps):
            if step.get('action') == 'loop':
                loop_steps = step.get('steps', [])
                if not loop_steps:
                    raise ValidationError(f"Loop step at index {i}: No steps defined in loop")
                
                # Recursively validate nested steps
                self._validate_loop_steps(loop_steps)
    
    def _read_file(self, filepath: str) -> str:
        """Read file from output directory or absolute path"""
        path = Path(filepath)
        
        # If not absolute, check in output directory first
        if not path.is_absolute():
            output_path = self.output_dir / path
            if output_path.exists():
                path = output_path
        
        try:
            return path.read_text(encoding='utf-8')
        except FileNotFoundError:
            raise ActionError(f"File not found: {filepath}")
        except Exception as e:
            raise ActionError(f"Error reading file {filepath}: {e}")
    
    def _create_backup(self, filename: str, content: str):
        """Create a timestamped backup of the file content"""
        try:
            # Ensure backup directory exists
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp in format: YYYYMMDD_HHMMSS
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Extract filename without path for backup naming
            base_filename = Path(filename).name
            
            # Create backup filename: DATETIME_originalfilename
            backup_filename = f"{timestamp}_{base_filename}"
            backup_path = self.backup_dir / backup_filename
            
            # Write backup
            backup_path.write_text(content, encoding='utf-8')
            logger.info(f"Created backup: {backup_path.relative_to(self.output_dir)}")
            
        except Exception as e:
            # Don't fail the main operation if backup fails, just warn
            logger.warning(f"Failed to create backup for '{filename}': {e}")
    
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
            base_url = "http://localhost:11434"
            try:
                client = ollama.Client()
                if hasattr(client, '_base_url'):
                    base_url = client._base_url
            except:
                pass
                
            import requests
            requests.post(
                f"{base_url}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=30
            )
            logger.debug(f"Successfully unloaded model: {model}")
        except Exception as e:
            logger.warning(f"Failed to unload model {model}: {e}")
    
    def _find_testenv_path(self) -> Path:
        """Find the testenv directory relative to the current working directory"""
        # Look for testenv in current directory first
        testenv_path = Path.cwd() / "testenv"
        if testenv_path.exists() and (testenv_path / "bin" / "activate").exists():
            return testenv_path
        
        # Look one level up
        testenv_path = Path.cwd().parent / "testenv"
        if testenv_path.exists() and (testenv_path / "bin" / "activate").exists():
            return testenv_path
        
        raise ActionError("testenv virtual environment not found. Expected testenv/bin/activate in current directory or parent directory.")
    
    async def _force_kill_python_process(self, command: str, pid: int):
        """Force kill Python process using pkill as last resort"""
        try:
            # Extract the script name from the command for targeted killing
            import shlex
            args = shlex.split(command)
            if len(args) > 1:
                script_name = Path(args[1]).name
                pkill_cmd = f"pkill -f python"
                logger.debug(f"Using pkill command: {pkill_cmd}")
                
                process = await asyncio.create_subprocess_shell(
                    pkill_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
        except Exception as e:
            logger.error(f"pkill fallback failed: {e}")
    
    async def execute(self):
        """Execute the scenario"""
        config = self.scenario.get('config', {})
        
        # Setup logging from config
        log_level = getattr(logging, config.get('logLevel', 'INFO').upper())
        logging.getLogger().setLevel(log_level)
        
        # Setup output directory and backup directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Backup directory: {self.backup_dir}")
        
        # Create agent instances
        await self._create_all_agents()
        
        # Execute workflow
        workflow_steps = self.scenario['workflow']
        
        logger.info(f"Starting scenario: {self.scenario.get('config', {}).get('name', 'Unnamed')}")
        
        try:
            await self._execute_steps(workflow_steps)
        finally:
            # Cleanup
            if self.current_model:
                self._unload_model(self.current_model)
    
    async def _create_all_agents(self):
        """Create all agent instances from definitions"""
        for agent_name, agent_def in self.scenario['agents'].items():
            agent = AgentInstance(
                name=agent_name,
                model=agent_def['model'],
                temperature=agent_def.get('temperature', 0.7),
                personality=agent_def.get('personality', ''),
                max_context_tokens=agent_def.get('maxContextTokens', 8000),
                query_timeout=agent_def.get('queryTimeout', 300),
            )
            self.agents[agent_name] = agent
            logger.info(f"Created agent: {agent_name} (model: {agent.model}, context: {agent.max_context_tokens} tokens, timeout: {agent.query_timeout}s)")
    
    def _substitute_loop_variables(self, step: Dict[str, Any], loop_context: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute loop variables in a step definition"""
        import copy
        step_copy = copy.deepcopy(step)
        
        # Substitute in all string values
        for key, value in step_copy.items():
            if isinstance(value, str):
                for var_name, var_value in loop_context.items():
                    value = value.replace(f'{{{{{var_name}}}}}', str(var_value))
                step_copy[key] = value
            elif isinstance(value, list):
                # Handle lists (like inputs)
                new_list = []
                for item in value:
                    if isinstance(item, str):
                        for var_name, var_value in loop_context.items():
                            item = item.replace(f'{{{{{var_name}}}}}', str(var_value))
                    new_list.append(item)
                step_copy[key] = new_list
            elif isinstance(value, dict):
                # Recursively handle nested dicts
                step_copy[key] = self._substitute_loop_variables(value, loop_context)
        
        return step_copy
    
    async def _execute_steps(self, steps: List[Dict[str, Any]], loop_context: Optional[Dict[str, Any]] = None):
        """Execute a list of steps"""
        for i, step in enumerate(steps):
            await self._execute_step(step, step_index=i, loop_context=loop_context)
    
    async def _execute_step(self, step: Dict[str, Any], step_index: int = 0, loop_context: Optional[Dict[str, Any]] = None):
        """Execute a single step"""
        step_id = step.get('id', f'step_{step_index}')
        
        # Process step to substitute loop variables
        if loop_context:
            step = self._substitute_loop_variables(step, loop_context)
        
        action = step['action']
        
        logger.info(f"Executing: {action} ({step_id})")
        
# Handle built-in actions
        if action == 'loop':
            await self._execute_loop(step, loop_context)
        elif action == 'clear_context':
            await self._clear_context(step)
        elif action == 'run_python':
            await self._execute_run_python(step, step_id)
        elif action in self.scenario['actions']:
            await self._execute_action(step, step_id, loop_context)
        else:
            raise ActionError(f"Unknown action: {action}")
    
    async def _execute_run_python(self, step: Dict[str, Any], step_id: str):
        """Execute a Python script in the testenv virtual environment on Ubuntu."""
        inputs = step.get('inputs', [])
        if not inputs:
            raise ActionError(f"run_python action '{step_id}' requires at least one input (the Python file to execute)")
        
        # Find testenv path
        try:
            testenv_path = self._find_testenv_path()
            python_executable = testenv_path / "bin" / "python"
            if not python_executable.exists():
                raise ActionError(f"Python executable not found in testenv at: {python_executable}")
        except ActionError as e:
            raise ActionError(f"run_python action '{step_id}': {e}")
            
        # Process inputs - read files or use text directly
        # Simple rule: if input contains a space, treat as literal text, otherwise treat as file
        # For run_python, we want the file path, not the content
        processed_inputs = []
        for input_value in inputs:
            if isinstance(input_value, str) and ' ' not in input_value:
                # Always resolve relative paths to absolute paths relative to output_dir
                path = Path(input_value)
                if not path.is_absolute():
                    absolute_path = (self.output_dir / path).resolve()
                    # It's good practice to ensure the file exists if it's meant to be a script
                    if not absolute_path.exists():
                        logger.warning(f"File '{input_value}' (resolved to '{absolute_path}') does not exist. Proceeding anyway.")
                    processed_inputs.append(str(absolute_path))
                else:
                    processed_inputs.append(str(path.resolve())) # Resolve absolute paths too for consistency
            else:
                # If it contains spaces, treat as a direct argument and ensure it's a string
                processed_inputs.append(str(input_value))
        
        # Construct the command using the absolute path to the Python executable
        # Quote each argument to handle spaces or special characters correctly.
        python_args = " ".join(f'"{arg}"' for arg in processed_inputs)
        command = f'"{python_executable}" {python_args}'
        
        logger.info(f"Executing Python script: {command}")
        
        try:
            # Execute with timeout
            # We are keeping shell=True because it was used before, but if you only run python scripts
            # and no other shell commands (like pipes, redirects), you might consider shell=False
            # and passing a list of arguments for better security.
            result = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Combine stderr with stdout
                cwd=self.output_dir  # Run from output directory
            )
            
            # Wait for completion with timeout
            try:
                stdout, _ = await asyncio.wait_for(result.communicate(), timeout=30.0)
                output = stdout.decode('utf-8', errors='replace')
                return_code = result.returncode
            except asyncio.TimeoutError:
                # Two-step process killing with escalating force
                logger.warning(f"Python script timed out after 30 seconds, attempting to kill process {result.pid}")
                
                # Step 1: Send SIGINT (Ctrl+C equivalent) - gentle shutdown
                try:
                    import signal
                    result.send_signal(signal.SIGINT)
                    logger.debug(f"Sent SIGINT to process {result.pid}")
                    
                    # Wait 3 seconds for graceful shutdown
                    try:
                        await asyncio.wait_for(result.wait(), timeout=3.0)
                        logger.info(f"Process {result.pid} exited gracefully after SIGINT")
                    except asyncio.TimeoutError:
                        # Step 2: Send SIGTERM (terminate) - more forceful
                        logger.debug(f"SIGINT failed, sending SIGTERM to process {result.pid}")
                        result.terminate()
                        
                        try:
                            await asyncio.wait_for(result.wait(), timeout=3.0)
                            logger.info(f"Process {result.pid} exited after SIGTERM")
                        except asyncio.TimeoutError:
                            # Step 3: Send SIGKILL (kill) - most forceful
                            logger.debug(f"SIGTERM failed, sending SIGKILL to process {result.pid}")
                            result.kill()
                            
                            try:
                                await asyncio.wait_for(result.wait(), timeout=2.0)
                                logger.info(f"Process {result.pid} killed with SIGKILL")
                            except asyncio.TimeoutError:
                                # Last resort: use pkill to find and kill by command pattern
                                logger.warning(f"SIGKILL failed, attempting pkill as last resort")
                                await self._force_kill_python_process(command, result.pid)
                                
                                # Final wait attempt
                                try:
                                    await asyncio.wait_for(result.wait(), timeout=1.0)
                                except asyncio.TimeoutError:
                                    logger.error(f"Failed to kill process {result.pid} completely")
                
                except Exception as e:
                    logger.error(f"Error during process killing: {e}")
                    # Fallback to original method
                    try:
                        result.kill()
                        await result.wait()
                    except:
                        pass
                
                output = "EXECUTION TIMED OUT AFTER 30 SECONDS\n"
                return_code = -1
            # Prepare output content
            output_content = f"=== Python Script Execution Results ===\n"
            output_content += f"Command: {command}\n"
            output_content += f"Return Code: {return_code}\n"
            output_content += f"Working Directory: {self.output_dir}\n"
            output_content += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            output_content += f"\n===   Output   ===\n"
            output_content += output
            output_content += f"\n=== End Output ===\n"

            
            if return_code != 0:
                output_content += f"\n=== Script exited with error code {return_code} ===\n"
            
            # Save output if specified
            if 'output' in step:
                await self._save_output(output_content, step['output'])
                logger.info(f"Python execution output saved to '{step['output']}'")
            
            if return_code == 0:
                logger.info(f"Python script executed successfully")
            else:
                logger.warning(f"Python script exited with code {return_code}")
                
        except Exception as e:
            error_output = f"=== Python Script Execution Error ===\n"
            error_output += f"Command: {command}\n"
            error_output += f"Error: {str(e)}\n"
            error_output += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            if 'output' in step:
                await self._save_output(error_output, step['output'])
            
            raise ActionError(f"Failed to execute Python script in step '{step_id}': {e}")
    
    async def _execute_action(self, step: Dict[str, Any], step_id: str, loop_context: Optional[Dict[str, Any]] = None):
        """Execute a user-defined action"""
        action_name = step['action']
        agent_name = step['agent']
        inputs = step.get('inputs', [])
        
        agent = self.agents[agent_name]
        action_def = self.scenario['actions'][action_name]
        prompt_template = action_def['prompt']
        
        # Ensure model is loaded
        self._ensure_model_loaded(agent.model)
        
        # Process inputs - read files or use text directly
        # Simple rule: if input contains a space, treat as literal text, otherwise treat as file
        processed_inputs = []
        for input_value in inputs:
            if isinstance(input_value, str) and ' ' not in input_value:
                try:
                    content = self._read_file(self.output_dir / input_value)
                    processed_inputs.append(content)
                    logger.debug(f"Read file '{input_value}': {len(content)} characters")
                except ActionError:
                    # If file doesn't exist, treat as plain text
                    processed_inputs.append(input_value)
                    logger.debug(f"File '{input_value}' not found, treating as literal text")
            else:
                processed_inputs.append(str(input_value))
                logger.debug(f"Using literal input: '{input_value}'")
        
        # Build prompt from template
        prompt = prompt_template
        for i, input_content in enumerate(processed_inputs, 1):
            placeholder = f'{{{{{i}}}}}'
            if placeholder in prompt:
                # Log substitution for debugging
                content_preview = input_content[:100] + "..." if len(input_content) > 100 else input_content
                logger.debug(f"Substituting {placeholder} with content: {content_preview}")
                prompt = prompt.replace(placeholder, input_content)
        
        # Execute query using agent-specific settings
        logger.info(f"Agent {agent_name} executing: {action_name}")
        logger.debug(f"Final prompt length: {len(prompt)} characters")
        result = await agent.query(prompt)
        
        # Save output if specified
        if 'output' in step:
            await self._save_output(result, step['output'], step.get('format'))
            logger.debug(f"Saved output to '{step['output']}': {len(result)} chars")
    
    async def _save_output(self, content: str, filename: str, file_format: Optional[str] = None):
        """Save content to file, with special handling for formats and automatic backup creation"""
        # Handle format-specific processing
        if file_format == "python":
            # Extract code from markdown block
            match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
            if match:
                content = match.group(1).strip()
                logger.info(f"Extracted Python code block from markdown for '{filename}'.")
            else:
                logger.warning(f"No Python code block found in content for '{filename}'. Saving raw content.")
            
            # Ensure .py extension
            if not filename.endswith(".py"):
                filename = f"{Path(filename).stem}.py"
                logger.info(f"Adjusted filename to '{filename}' for Python format.")
                
        # Create backup before saving new content
        self._create_backup(filename, content)
        
        # Save the main file
        output_path = self.output_dir / filename
        output_path.write_text(content, encoding='utf-8')
        logger.info(f"Saved output to: {output_path}")
    
    async def _execute_loop(self, step: Dict[str, Any], parent_context: Optional[Dict[str, Any]] = None):
        """Execute a loop"""
        iterations = step['iterations']
        loop_steps = step['steps']
        
        for i in range(iterations):
            logger.info(f"Loop iteration {i + 1}/{iterations}")
            
            # Create context with iteration variable
            loop_context = {
                'iteration': i + 1
            }
            if parent_context:
                loop_context.update(parent_context)
            
            await self._execute_steps(loop_steps, loop_context)
    
    async def _clear_context(self, step: Dict[str, Any]):
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
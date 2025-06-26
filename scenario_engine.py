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
import jsonschema

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
        """
        Rough estimation of token count.
        Note: This is a very rough approximation and may be inaccurate for code or non-English text.
        """
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
            current_prompt = other_messages.pop() # Use pop() for efficiency
            other_messages = other_messages

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

    def __init__(self, scenario_path: str, loaded_scenario: Dict[str, Any]):
        self.scenario_path = scenario_path
        self.scenario = loaded_scenario
        
        # Get working directory and other configs
        config = self.scenario.get('config', {})
        self.output_dir = Path(config.get('workDir', './results'))
        self.backup_dir = self.output_dir / '.backup'
        
        # --- MODIFICATION: Add configurable python timeout ---
        self.python_timeout = float(config.get('pythonTimeout', 30.0))
        
        # --- MODIFICATION: Create internal maps from lists instead of relying on mutated data ---
        # The executor now works with the original list structure defined in the JSON file.
        self.agents_map: Dict[str, Dict] = {agent['name']: agent for agent in self.scenario.get('agents', [])}
        self.actions_map: Dict[str, Dict] = {action['name']: action for action in self.scenario.get('actions', [])}

        # This dictionary holds the RUNTIME agent instances
        self.agents: Dict[str, AgentInstance] = {}
        
        self.current_model: Optional[str] = None

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
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = Path(filename).name
            backup_filename = f"{timestamp}_{base_filename}"
            backup_path = self.backup_dir / backup_filename

            backup_path.write_text(content, encoding='utf-8')
            logger.info(f"Created backup: {backup_path.relative_to(self.output_dir)}")
            
        except Exception as e:
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
        current_path = Path.cwd()
        for i in range(3): # Check current dir and two parents
            testenv_path = current_path / "testenv"
            if testenv_path.exists() and (testenv_path / "bin" / "activate").exists():
                return testenv_path
            if current_path.parent == current_path: # Reached root
                break
            current_path = current_path.parent
        
        raise ActionError("testenv virtual environment not found. Expected testenv/bin/activate in project hierarchy.")

    async def _force_kill_python_process(self, command: str, pid: int):
        """Force kill Python process using pkill as last resort"""
        try:
            # Extract the script name from the command for targeted killing
            import shlex
            args = shlex.split(command)
            if len(args) > 1:
                # Make pkill more specific to avoid killing unrelated processes
                script_name = Path(args[1]).name
                pkill_cmd = f"pkill -f 'python.*{script_name}'"
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
        
        log_level = getattr(logging, config.get('logLevel', 'INFO').upper())
        logging.getLogger().setLevel(log_level)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Backup directory: {self.backup_dir}")
        logger.info(f"Python execution timeout set to: {self.python_timeout} seconds")
        
        # Create agent instances
        await self._create_all_agents()
        
        workflow_steps = self.scenario['workflow']
        logger.info(f"Starting scenario: {self.scenario.get('config', {}).get('name', 'Unnamed')}")
        
        try:
            await self._execute_steps(workflow_steps)
        finally:
            # Cleanup
            if self.current_model:
                self._unload_model(self.current_model)

    async def _create_all_agents(self):
        """Create all agent instances from definitions using the agents_map."""
        # --- MODIFICATION: Use self.agents_map which was built from the list ---
        for agent_name, agent_def in self.agents_map.items():
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
        
        for key, value in step_copy.items():
            if isinstance(value, str):
                for var_name, var_value in loop_context.items():
                    value = value.replace(f'{{{{{var_name}}}}}', str(var_value))
                step_copy[key] = value
            elif isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, str):
                        for var_name, var_value in loop_context.items():
                            item = item.replace(f'{{{{{var_name}}}}}', str(var_value))
                    new_list.append(item)
                step_copy[key] = new_list
            elif isinstance(value, dict):
                step_copy[key] = self._substitute_loop_variables(value, loop_context)
        
        return step_copy

    async def _execute_steps(self, steps: List[Dict[str, Any]], loop_context: Optional[Dict[str, Any]] = None):
        """Execute a list of steps"""
        for i, step in enumerate(steps):
            await self._execute_step(step, step_index=i, loop_context=loop_context)

    async def _execute_step(self, step: Dict[str, Any], step_index: int = 0, loop_context: Optional[Dict[str, Any]] = None):
        """Execute a single step"""
        step_id = step.get('id', f'step_{step_index}')
        
        if loop_context:
            step = self._substitute_loop_variables(step, loop_context)
        
        action = step['action']
        
        logger.info(f"Executing: {action} ({step_id})")
        
        # --- MODIFICATION: Check for action existence in the actions_map ---
        if action == 'loop':
            await self._execute_loop(step, loop_context)
        elif action == 'clear_context':
            await self._clear_context(step)
        elif action == 'run_python':
            await self._execute_run_python(step, step_id)
        elif action in self.actions_map:
            await self._execute_action(step, step_id, loop_context)
        else:
            raise ActionError(f"Unknown action: {action}")

    async def _execute_run_python(self, step: Dict[str, Any], step_id: str):
        """Execute a Python script in the testenv virtual environment on Ubuntu."""
        inputs = step.get('inputs', [])
        if not inputs:
            raise ActionError(f"run_python action '{step_id}' requires at least one input (the Python file to execute)")
        
        try:
            testenv_path = self._find_testenv_path()
            python_executable = testenv_path / "bin" / "python"
            if not python_executable.exists():
                raise ActionError(f"Python executable not found in testenv at: {python_executable}")
        except ActionError as e:
            raise ActionError(f"run_python action '{step_id}': {e}")
            
        processed_inputs = []
        for input_value in inputs:
            if isinstance(input_value, str) and ' ' not in input_value:
                path = Path(input_value)
                if not path.is_absolute():
                    absolute_path = (self.output_dir / path).resolve()
                    if not absolute_path.exists():
                        logger.warning(f"File '{input_value}' (resolved to '{absolute_path}') does not exist. Proceeding anyway.")
                    processed_inputs.append(str(absolute_path))
                else:
                    processed_inputs.append(str(path.resolve()))
            else:
                processed_inputs.append(str(input_value))
        
        # --- SECURITY FIX: Use list of arguments instead of shell=True ---
        command_parts = [str(python_executable)] + processed_inputs
        logger.info(f"Executing Python script: {command_parts}")

        try:
            # Execute without shell=True to prevent command injection
            result = await asyncio.create_subprocess_exec(
                *command_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.output_dir
            )
            
            # --- MODIFICATION: Use configurable timeout ---
            try:
                stdout, _ = await asyncio.wait_for(result.communicate(), timeout=self.python_timeout)
                output = stdout.decode('utf-8', errors='replace')
                return_code = result.returncode
            except asyncio.TimeoutError:
                logger.warning(f"Python script timed out after {self.python_timeout} seconds, attempting to kill process {result.pid}")
                
                try:
                    import signal
                    result.send_signal(signal.SIGINT)
                    logger.debug(f"Sent SIGINT to process {result.pid}")
                    
                    try:
                        await asyncio.wait_for(result.wait(), timeout=3.0)
                        logger.info(f"Process {result.pid} exited gracefully after SIGINT")
                    except asyncio.TimeoutError:
                        logger.debug(f"SIGINT failed, sending SIGTERM to process {result.pid}")
                        result.terminate()
                        
                        try:
                            await asyncio.wait_for(result.wait(), timeout=3.0)
                            logger.info(f"Process {result.pid} exited after SIGTERM")
                        except asyncio.TimeoutError:
                            logger.debug(f"SIGTERM failed, sending SIGKILL to process {result.pid}")
                            result.kill()
                            
                            try:
                                await asyncio.wait_for(result.wait(), timeout=2.0)
                                logger.info(f"Process {result.pid} killed with SIGKILL")
                            except asyncio.TimeoutError:
                                logger.error(f"Failed to kill process {result.pid} completely")
                
                except Exception as e:
                    logger.error(f"Error during process killing: {e}")
                    try:
                        result.kill()
                        await result.wait()
                    except:
                        pass
                
                output = f"EXECUTION TIMED OUT AFTER {self.python_timeout} SECONDS\n"
                return_code = -1
            
            output_content = f"=== Python Script Execution Results ===\n"
            output_content += f"Command: {' '.join(command_parts)}\n"
            output_content += f"Return Code: {return_code}\n"
            output_content += f"Working Directory: {self.output_dir}\n"
            output_content += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            output_content += f"\n===   Output   ===\n"
            output_content += output
            output_content += f"\n=== End Output ===\n"

            if return_code != 0:
                output_content += f"\n=== Script exited with error code {return_code} ===\n"
            
            if 'output' in step:
                await self._save_output(output_content, step['output'])
                logger.info(f"Python execution output saved to '{step['output']}'")
            
            if return_code == 0:
                logger.info(f"Python script executed successfully")
            else:
                logger.warning(f"Python script exited with code {return_code}")
                
        except Exception as e:
            error_output = f"=== Python Script Execution Error ===\n"
            error_output += f"Command: {' '.join(command_parts)}\n"
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
        # --- MODIFICATION: Get action definition from the actions_map ---
        action_def = self.actions_map[action_name]
        prompt_template = action_def['prompt']
        
        self._ensure_model_loaded(agent.model)
        
        processed_inputs = []
        for input_value in inputs:
            if isinstance(input_value, str) and ' ' not in input_value:
                try:
                    content = self._read_file(self.output_dir / input_value)
                    processed_inputs.append(content)
                    logger.debug(f"Read file '{input_value}': {len(content)} characters")
                except ActionError:
                    processed_inputs.append(input_value)
                    logger.debug(f"File '{input_value}' not found, treating as literal text")
            else:
                processed_inputs.append(str(input_value))
                logger.debug(f"Using literal input: '{input_value}'")
        
        prompt = prompt_template
        for i, input_content in enumerate(processed_inputs, 1):
            placeholder = f'{{{{{i}}}}}'
            if placeholder in prompt:
                content_preview = (str(input_content)[:100] + "...") if len(str(input_content)) > 100 else str(input_content)
                logger.debug(f"Substituting {placeholder} with content: {content_preview}")
                prompt = prompt.replace(placeholder, str(input_content))
        
        logger.info(f"Agent {agent_name} executing: {action_name}")
        logger.debug(f"Final prompt length: {len(prompt)} characters")
        result = await agent.query(prompt)
        
        if 'output' in step:
            await self._save_output(result, step['output'], step.get('format'))
            logger.debug(f"Saved output to '{step['output']}': {len(result)} chars")

    async def _save_output(self, content: str, filename: str, file_format: Optional[str] = None):
        """Save content to file, with special handling for formats and automatic backup creation"""
        if file_format == "python":
            match = re.search(r"```(?:python)?\n(.*?)\n```", content, re.DOTALL)
            if match:
                content = match.group(1).strip()
                logger.info(f"Extracted Python code block from markdown for '{filename}'.")
            else:
                logger.warning(f"No Python code block found in content for '{filename}'. Saving raw content.")
            
            if not filename.endswith(".py"):
                filename = f"{Path(filename).stem}.py"
                logger.info(f"Adjusted filename to '{filename}' for Python format.")
                
        # Create a backup of the previous file content if it exists
        output_path = self.output_dir / filename
        if output_path.exists():
            previous_content = output_path.read_text(encoding='utf-8')
            self._create_backup(filename, previous_content)
        
        output_path.write_text(content, encoding='utf-8')
        logger.info(f"Saved output to: {output_path}")

    async def _execute_loop(self, step: Dict[str, Any], parent_context: Optional[Dict[str, Any]] = None):
        """Execute a loop"""
        iterations = step['iterations']
        loop_steps = step['steps']
        
        for i in range(iterations):
            logger.info(f"Loop iteration {i + 1}/{iterations}")
            
            loop_context = {'iteration': i + 1}
            if parent_context:
                loop_context.update(parent_context)
            
            await self._execute_steps(loop_steps, loop_context)

    async def _clear_context(self, step: Dict[str, Any]):
        """Clear conversation history for specified agent"""
        agent_name = step.get('agent')
        
        if not agent_name:
            raise ActionError("clearContext requires an 'agent' field")
        
        # Accessing self.agents is correct here, as it's the dictionary of runtime instances
        if agent_name not in self.agents:
            raise ActionError(f"Unknown agent: {agent_name}")
        
        self.agents[agent_name].clear_context()
        logger.info(f"Cleared context for agent: {agent_name}")

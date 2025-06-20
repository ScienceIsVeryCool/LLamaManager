#!/usr/bin/env python3
"""
Ollama Multi-Agent Conversation Manager
Supports dual-agent debate mode with mantras
"""

import asyncio
import json
import logging
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import ollama
from datetime import datetime
import yaml


# Constants
DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_LOG_DIR = "log"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_ROUNDS = 3
OLLAMA_API_URL = "http://localhost:11434/api/generate"
PREVIEW_LENGTH = 200

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OllamaAgentError(Exception):
    """Base exception for Ollama Agent errors"""
    pass


class ConfigurationError(OllamaAgentError):
    """Raised when configuration is invalid"""
    pass


class ModelError(OllamaAgentError):
    """Raised when model operations fail"""
    pass


@dataclass
class Message:
    role: str
    content: str
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls"""
        return {"role": self.role, "content": self.content}


@dataclass
class TaskResult:
    """Structured result for task execution"""
    task: str
    mode: str
    agent: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    final_solution: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Agent:
    """Single agent that can interact with Ollama"""
    
    def __init__(self, name: str, model: str, system_prompt: str = "", 
                 mantra: str = "", temperature: float = DEFAULT_TEMPERATURE):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.mantra = mantra
        self.temperature = temperature
        self.conversation_history: List[Message] = []
        self._client: Optional[ollama.Client] = None
        logger.info(f"Initialized agent '{name}' with model '{model}'")
    
    @property
    def client(self) -> ollama.Client:
        """Lazy initialization of Ollama client"""
        if self._client is None:
            try:
                self._client = ollama.Client()
            except Exception as e:
                raise ModelError(f"Failed to initialize Ollama client: {e}")
        return self._client
    
    def _build_messages(self, prompt: str, include_history: bool = True) -> List[Dict[str, str]]:
        """Build message list for API call"""
        messages = []
        
        # Add system prompt if exists
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add conversation history if requested
        if include_history:
            messages.extend([msg.to_dict() for msg in self.conversation_history])
        
        # Combine prompt with mantra if exists
        final_prompt = self._apply_mantra(prompt)
        messages.append({"role": "user", "content": final_prompt})
        
        return messages
    
    def _apply_mantra(self, prompt: str) -> str:
        """Apply mantra to prompt if available"""
        if self.mantra:
            return f"{prompt}\n\n[Remember: {self.mantra}]"
        return prompt
    
    async def query(self, prompt: str, include_history: bool = True) -> str:
        """Send a query to the model and get response"""
        try:
            messages = self._build_messages(prompt, include_history)
            
            logger.debug(f"Querying {self.name} with {len(messages)} messages")
            
            # Query Ollama
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": self.temperature}
            )
            
            response_content = response['message']['content']
            
            # Store in history (without mantra for cleaner logs)
            self.conversation_history.extend([
                Message("user", prompt),
                Message("assistant", response_content)
            ])
            
            logger.debug(f"Received response from {self.name}: {len(response_content)} characters")
            return response_content
            
        except Exception as e:
            raise ModelError(f"Query failed for agent {self.name}: {e}")
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        logger.debug(f"Cleared history for agent {self.name}")


class ConfigValidator:
    """Validates configuration files"""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate configuration structure"""
        required_keys = ['agents']
        for key in required_keys:
            if key not in config:
                raise ConfigurationError(f"Missing required config key: {key}")
        
        if not config['agents']:
            raise ConfigurationError("No agents defined in configuration")
        
        for i, agent_config in enumerate(config['agents']):
            ConfigValidator._validate_agent_config(agent_config, i)
        
        # Validate debate config if present
        if 'debate' in config:
            ConfigValidator._validate_debate_config(config['debate'], config['agents'])
    
    @staticmethod
    def _validate_agent_config(agent_config: Dict[str, Any], index: int) -> None:
        """Validate individual agent configuration"""
        required_keys = ['name', 'model']
        for key in required_keys:
            if key not in agent_config:
                raise ConfigurationError(f"Agent {index} missing required key: {key}")
        
        # Validate temperature if present
        if 'temperature' in agent_config:
            temp = agent_config['temperature']
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                raise ConfigurationError(f"Invalid temperature for agent {index}: {temp}")
    
    @staticmethod
    def _validate_debate_config(debate_config: Dict[str, Any], agents: List[Dict[str, Any]]) -> None:
        """Validate debate configuration"""
        required_keys = ['creator', 'judge']
        agent_names = {agent['name'] for agent in agents}
        
        for key in required_keys:
            if key not in debate_config:
                raise ConfigurationError(f"Debate config missing required key: {key}")
            
            if debate_config[key] not in agent_names:
                raise ConfigurationError(f"Debate {key} '{debate_config[key]}' not found in agents")


class FileManager:
    """Handles file operations"""
    
    @staticmethod
    def ensure_directory(path: str) -> None:
        """Ensure directory exists"""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def save_revision(content: str, task: str, feedback: str, round_num: int, 
                     log_dir: str = DEFAULT_LOG_DIR) -> None:
        """Save a revision to file"""
        FileManager.ensure_directory(log_dir)
        filename = Path(log_dir) / f"revision_{round_num}.md"
        
        with open(filename, "w", encoding='utf-8') as file:
            file.write(f"# Revision {round_num}\n\n")
            file.write(f"## Task\n{task}\n\n")
            file.write(f"## Feedback\n{feedback}\n\n")
            file.write(f"## Revised Solution\n{content}")
        
        logger.info(f"Saved revision {round_num} to {filename}")


class ModelManager:
    """Handles model lifecycle operations"""
    
    @staticmethod
    def unload_model(model_name: str) -> None:
        """Explicitly unload a model from memory"""
        try:
            import requests
            response = requests.post(
                OLLAMA_API_URL,
                json={"model": model_name, "keep_alive": 0},
                timeout=30
            )
            response.raise_for_status()
            logger.info(f"Unloaded model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to unload model {model_name}: {e}")
    
    @staticmethod
    def unload_models(model_names: List[str]) -> None:
        """Unload multiple models"""
        for model_name in model_names:
            ModelManager.unload_model(model_name)


class ConversationManager:
    """Manages multi-agent conversations"""
    
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.config = self._load_config()
        self.agents: Dict[str, Agent] = {}
        self._setup_agents()
        logger.info(f"Initialized ConversationManager with {len(self.agents)} agents")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            ConfigValidator.validate_config(config)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
            
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration: {e}")
    
    def _setup_agents(self) -> None:
        """Initialize agents from config"""
        for agent_config in self.config['agents']:
            agent = Agent(
                name=agent_config['name'],
                model=agent_config['model'],
                system_prompt=agent_config.get('system_prompt', ''),
                mantra=agent_config.get('mantra', ''),
                temperature=agent_config.get('temperature', DEFAULT_TEMPERATURE)
            )
            self.agents[agent.name] = agent
    
    def _get_debate_config(self) -> Dict[str, Any]:
        """Get debate configuration with defaults"""
        debate_config = self.config.get('debate', {})
        return {
            'creator': debate_config.get('creator'),
            'judge': debate_config.get('judge'),
            'rounds': debate_config.get('rounds', DEFAULT_ROUNDS)
        }
    
    async def _create_initial_solution(self, creator: Agent, task: str) -> str:
        """Create initial solution"""
        logger.info(f"[{creator.name}] Creating initial response...")
        response = await creator.query(
            f"Task: {task}\n\nPlease provide your best complete solution."
        )
        logger.info(f"Initial response: {response[:PREVIEW_LENGTH]}...")
        return response
    
    async def _evaluate_solution(self, judge: Agent, task: str, solution: str) -> str:
        """Evaluate a solution"""
        logger.info(f"[{judge.name}] Evaluating...")
        evaluation = await judge.query(
            f"Task: {task}\n\n"
            f"Solution provided:\n{solution}\n\n"
            f"Evaluate this solution. What are its strengths and weaknesses? "
            f"What specific improvements would you suggest?"
        )
        logger.info(f"Evaluation: {evaluation[:PREVIEW_LENGTH]}...")
        return evaluation
    
    async def _revise_solution(self, creator: Agent, task: str, feedback: str) -> str:
        """Revise solution based on feedback"""
        logger.info(f"[{creator.name}] Revising based on feedback...")
        revision = await creator.query(
            f"Task: {task}\n\n"
            f"Feedback received:\n{feedback}\n\n"
            f"Please provide a complete revised solution addressing the feedback."
        )
        logger.info(f"Revision: {revision[:PREVIEW_LENGTH]}...")
        return revision
    
    async def debate_task(self, task: str, save_revisions: bool = True) -> TaskResult:
        """Execute a task with creator-judge debate"""
        debate_config = self._get_debate_config()
        
        if not debate_config['creator'] or not debate_config['judge']:
            raise ConfigurationError("Debate mode requires 'creator' and 'judge' agents in config")
        
        creator = self.agents[debate_config['creator']]
        judge = self.agents[debate_config['judge']]
        rounds = debate_config['rounds']
        
        results = []
        
        # Creator's initial attempt
        creator_response = await self._create_initial_solution(creator, task)
        results.append({
            "round": 0,
            "agent": creator.name,
            "type": "creation",
            "content": creator_response
        })
        
        # Debate rounds
        for round_num in range(rounds):
            logger.info(f"\n--- Round {round_num + 1} ---")
            
            # Judge evaluates
            judge_response = await self._evaluate_solution(judge, task, creator_response)
            results.append({
                "round": round_num + 1,
                "agent": judge.name,
                "type": "evaluation",
                "content": judge_response
            })
            
            # Creator revises based on feedback
            creator_response = await self._revise_solution(creator, task, judge_response)
            results.append({
                "round": round_num + 1,
                "agent": creator.name,
                "type": "revision",
                "content": creator_response
            })
            
            # Save revision if requested
            if save_revisions:
                FileManager.save_revision(creator_response, task, judge_response, round_num + 1)
        
        return TaskResult(
            task=task,
            mode="debate",
            results=results,
            final_solution=creator_response
        )
    
    async def self_reflection_task(self, task: str, agent_name: Optional[str] = None) -> TaskResult:
        """Execute a task with self-reflection"""
        if agent_name is None:
            agent_name = self.config.get('default_agent')
            if not agent_name:
                raise ConfigurationError("No agent specified and no default_agent in config")
        
        if agent_name not in self.agents:
            raise ConfigurationError(f"Agent '{agent_name}' not found")
        
        agent = self.agents[agent_name]
        results = []
        
        # Initial response
        logger.info(f"[{agent.name}] Working on task...")
        response = await agent.query(task)
        results.append({
            "iteration": 0,
            "type": "initial",
            "content": response
        })
        
        # Simple self-reflection
        reflection_prompt = "Review your previous response. What could be improved? Provide an updated solution."
        logger.info(f"[{agent.name}] Self-reflecting...")
        reflection = await agent.query(reflection_prompt)
        results.append({
            "iteration": 1,
            "type": "reflection",
            "content": reflection
        })
        
        return TaskResult(
            task=task,
            mode="self_reflection",
            agent=agent_name,
            results=results,
            final_solution=reflection
        )
    
    async def run_task(self, task: str, mode: str = "self_reflection") -> TaskResult:
        """Run a task in specified mode"""
        if mode == "self_reflection":
            return await self.self_reflection_task(task)
        elif mode == "debate":
            return await self.debate_task(task)
        else:
            raise ValueError(f"Unknown mode: {mode}. Supported modes: self_reflection, debate")
    
    def unload_all_models(self) -> None:
        """Unload all models used by agents"""
        model_names = [agent.model for agent in self.agents.values()]
        ModelManager.unload_models(list(set(model_names)))  # Remove duplicates
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        logger.info("Cleaning up ConversationManager resources")
        self.unload_all_models()


async def main():
    """Example usage"""
    try:
        FileManager.ensure_directory(DEFAULT_LOG_DIR)
        
        manager = ConversationManager()
        
        # Example task
        task = "Write a Python function to calculate the fibonacci sequence efficiently"
        
        # Run debate
        logger.info("=== DEBATE MODE ===")
        result = await manager.run_task(task, mode="debate")
        
        # Save results
        output_file = "results_debate.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")
        
        # Cleanup
        manager.cleanup()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
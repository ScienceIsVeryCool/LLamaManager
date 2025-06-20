#!/usr/bin/env python3
"""
Ollama Multi-Agent Conversation Manager
Supports self-reflection and dual-agent debate modes
"""

import asyncio
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import ollama
from datetime import datetime
import yaml


@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class Agent:
    """Single agent that can interact with Ollama"""
    
    def __init__(self, name: str, model: str, system_prompt: str = "", temperature: float = 0.7):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.conversation_history: List[Message] = []
        self.client = ollama.Client()
    
    async def query(self, prompt: str, include_history: bool = True) -> str:
        """Send a query to the model and get response"""
        messages = []
        
        # Add system prompt if exists
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add conversation history if requested
        if include_history:
            for msg in self.conversation_history:
                messages.append({"role": msg.role, "content": msg.content})
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        # Query Ollama
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature}
        )
        
        # Store in history
        self.conversation_history.append(Message("user", prompt))
        self.conversation_history.append(Message("assistant", response['message']['content']))
        
        return response['message']['content']
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


class ConversationManager:
    """Manages single or multi-agent conversations"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.agents = {}
        self._setup_agents()
    
    def _setup_agents(self):
        """Initialize agents from config"""
        for agent_config in self.config['agents']:
            agent = Agent(
                name=agent_config['name'],
                model=agent_config['model'],
                system_prompt=agent_config.get('system_prompt', ''),
                temperature=agent_config.get('temperature', 0.7)
            )
            self.agents[agent_config['name']] = agent
    
    async def self_reflection_task(self, task: str, agent_name: str = None) -> Dict:
        """Execute a task with self-reflection"""
        if agent_name is None:
            agent_name = self.config['default_agent']
        
        agent = self.agents[agent_name]
        results = []
        
        # Initial response
        print(f"\n[{agent.name}] Working on task...")
        initial_response = await agent.query(task)
        results.append({
            "iteration": 0,
            "type": "initial",
            "content": initial_response
        })
        print(f"Initial response: {initial_response}...")
        
        # Self-reflection iterations
        reflection_prompts = self.config['reflection_prompts']
        for i, reflection_prompt in enumerate(reflection_prompts):
            print(f"\n[{agent.name}] Reflecting (iteration {i+1})...")
            reflection = await agent.query(reflection_prompt)
            results.append({
                "iteration": i + 1,
                "type": "reflection",
                "content": reflection
            })
            print(f"Reflection {i+1}: {reflection}...")
        
        return {
            "task": task,
            "agent": agent_name,
            "results": results
        }
    
    async def debate_task(self, task: str) -> Dict:
        """Execute a task with creator-judge debate"""
        creator_name = self.config['debate']['creator']
        judge_name = self.config['debate']['judge']
        rounds = self.config['debate']['rounds']
        
        creator = self.agents[creator_name]
        judge = self.agents[judge_name]
        
        results = []
        
        # Creator's initial attempt
        print(f"\n[{creator.name}] Creating initial response...")
        creator_prompt = f"Task: {task}\n\nPlease provide your best solution."
        creator_response = await creator.query(creator_prompt)
        results.append({
            "round": 0,
            "agent": creator_name,
            "type": "creation",
            "content": creator_response
        })
        print(f"Creator: {creator_response[:200]}...")
        
        # Debate rounds
        for round_num in range(rounds):
            print(f"\n--- Round {round_num + 1} ---")
            
            # Judge evaluates
            print(f"[{judge.name}] Evaluating...")
            judge_prompt = f"""Task: {task}

Solution provided:
{creator_response}

Please evaluate this solution. What are its strengths and weaknesses? 
What improvements would you suggest? Be specific and constructive."""
            
            judge_response = await judge.query(judge_prompt)
            results.append({
                "round": round_num + 1,
                "agent": judge_name,
                "type": "evaluation",
                "content": judge_response
            })
            print(f"Judge: {judge_response[:200]}...")
            
            # Creator revises based on feedback
            print(f"[{creator.name}] Revising based on feedback...")
            creator_prompt = f"""Based on this feedback:
{judge_response}

Please revise your solution to address the concerns raised."""
            
            creator_response = await creator.query(creator_prompt)
            results.append({
                "round": round_num + 1,
                "agent": creator_name,
                "type": "revision",
                "content": creator_response
            })
            print(f"Creator revision: {creator_response[:200]}...")
        
        return {
            "task": task,
            "mode": "debate",
            "results": results
        }
    
    async def run_task(self, task: str, mode: str = "self_reflection") -> Dict:
        """Run a task in specified mode"""
        if mode == "self_reflection":
            return await self.self_reflection_task(task)
        elif mode == "debate":
            return await self.debate_task(task)
        else:
            raise ValueError(f"Unknown mode: {mode}")


async def main():
    """Example usage"""
    manager = ConversationManager()
    
    # Example task
    task = "Write a Python function to calculate the fibonacci sequence"
    
    # Run in self-reflection mode
    print("=== SELF-REFLECTION MODE ===")
    result = await manager.run_task(task, mode="self_reflection")
    
    # Save results
    with open("results_self_reflection.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    # Run in debate mode
    print("\n\n=== DEBATE MODE ===")
    result = await manager.run_task(task, mode="debate")
    
    # Save results
    with open("results_debate.json", "w") as f:
        json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    asyncio.run(main())
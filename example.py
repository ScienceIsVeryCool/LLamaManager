#!/usr/bin/env python3
"""
Example usage of the Ollama Agent Manager
"""

import asyncio
from ollama_agent import ConversationManager


async def run_examples():
    # Initialize the manager
    manager = ConversationManager()
    
    print("=== Ollama Agent Manager Examples ===\n")
    
    # Example 1: Simple self-reflection
    print("1. Self-Reflection Example")
    print("-" * 40)
    task1 = "Write a Python function to check if a number is prime"
    result1 = await manager.run_task(task1, mode="self_reflection")
    print(f"Task completed with {len(result1['results'])} iterations\n")
    
    # Example 2: Debate mode
    print("2. Debate Mode Example")
    print("-" * 40)
    task2 = "Design a simple cache implementation with TTL support"
    result2 = await manager.run_task(task2, mode="debate")
    print(f"Debate completed with {len(result2['results'])} exchanges\n")
    
    # Example 3: Direct agent usage
    print("3. Direct Agent Usage Example")
    print("-" * 40)
    agent = manager.agents['thinker']
    
    # First query
    response1 = await agent.query("What are the benefits of async programming?")
    print(f"First response: {response1[:150]}...")
    
    # Follow-up with context
    response2 = await agent.query("Can you provide a simple example?")
    print(f"Follow-up: {response2[:150]}...")
    
    # Clear history and start fresh
    agent.clear_history()
    response3 = await agent.query("What is recursion?")
    print(f"New topic: {response3[:150]}...")


if __name__ == "__main__":
    asyncio.run(run_examples())
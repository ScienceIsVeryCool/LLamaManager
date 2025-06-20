#!/usr/bin/env python3
"""
Simple CLI for Ollama Agent Manager
"""

import asyncio
import argparse
import json
from ollama_agent import ConversationManager


async def main():
    print("Hello")
    parser = argparse.ArgumentParser(description='Ollama Multi-Agent Task Manager')
    parser.add_argument('task', help='The task to execute')
    parser.add_argument('--mode', choices=['self_reflection', 'debate'], 
                       default='self_reflection', help='Execution mode')
    parser.add_argument('--config', default='config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--output', help='Output file for results (JSON)')
    parser.add_argument('--agent', help='Agent name for self-reflection mode')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = ConversationManager(args.config)
    
    # Run task
    if args.mode == 'self_reflection' and args.agent:
        result = await manager.self_reflection_task(args.task, args.agent)
    else:
        result = await manager.run_task(args.task, args.mode)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")
    else:
        print("\n=== FINAL RESULTS ===")
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
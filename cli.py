#!/usr/bin/env python3
"""
Simple CLI for Ollama Agent Manager
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from ollama_agent import ConversationManager, OllamaAgentError, TaskResult


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def save_results(result: TaskResult, output_path: str) -> None:
    """Save results to file with proper error handling"""
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        print(f"\nResults saved to {output_path}")
        
    except Exception as e:
        print(f"Error saving results to {output_path}: {e}", file=sys.stderr)


def print_results(result: TaskResult) -> None:
    """Print results in a formatted way"""
    print("\n" + "="*50)
    print("TASK EXECUTION COMPLETE")
    print("="*50)
    print(f"Task: {result.task}")
    print(f"Mode: {result.mode}")
    if result.agent:
        print(f"Agent: {result.agent}")
    
    if result.final_solution:
        print(f"\nFinal Solution:\n{'-'*20}")
        print(result.final_solution)
    
    print("\n" + "="*50)


async def run_task_with_cleanup(manager: ConversationManager, task: str, 
                              mode: str, agent: str = None) -> TaskResult:
    """Run task with proper cleanup"""
    try:
        if mode == 'self_reflection' and agent:
            result = await manager.self_reflection_task(task, agent)
        else:
            result = await manager.run_task(task, mode)
        return result
    finally:
        # Always cleanup, even if task fails
        manager.cleanup()


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description='Ollama Multi-Agent Task Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Write a Python sorting algorithm" --mode debate
  %(prog)s "Explain quantum computing" --mode self_reflection --agent researcher
  %(prog)s "Design a web API" --mode debate --output results.json
        """
    )
    
    parser.add_argument('task', 
                       help='The task to execute')
    
    parser.add_argument('--mode', 
                       choices=['self_reflection', 'debate'], 
                       default='self_reflection', 
                       help='Execution mode (default: %(default)s)')
    
    parser.add_argument('--config', 
                       default='config.yaml', 
                       help='Path to configuration file (default: %(default)s)')
    
    parser.add_argument('--output', 
                       help='Output file for results (JSON format)')
    
    parser.add_argument('--agent', 
                       help='Agent name for self-reflection mode')
    
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Enable verbose logging')
    
    parser.add_argument('--no-cleanup', 
                       action='store_true',
                       help='Skip model cleanup after execution')
    
    return parser


async def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate arguments
        if args.mode == 'self_reflection' and args.agent:
            logger.info(f"Running self-reflection mode with agent: {args.agent}")
        elif args.mode == 'debate':
            logger.info("Running debate mode")
        else:
            logger.info("Running self-reflection mode with default agent")
        
        # Initialize manager
        logger.info(f"Loading configuration from: {args.config}")
        manager = ConversationManager(args.config)
        
        # Run task
        logger.info(f"Executing task: {args.task[:100]}...")
        result = await run_task_with_cleanup(manager, args.task, args.mode, args.agent)
        
        # Handle no-cleanup flag
        if not args.no_cleanup:
            logger.info("Cleanup completed")
        else:
            logger.info("Skipping cleanup (--no-cleanup flag set)")
        
        # Output results
        if args.output:
            save_results(result, args.output)
        else:
            print_results(result)
        
        logger.info("Task execution completed successfully")
        
    except OllamaAgentError as e:
        print(f"Agent Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    except FileNotFoundError as e:
        print(f"File Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(130)
        
    except Exception as e:
        logger.exception("Unexpected error occurred")
        print(f"Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
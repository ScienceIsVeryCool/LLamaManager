#!/usr/bin/env python3
"""
CLI for Simplified Scenario-based Ollama Agent Manager
"""

import asyncio
import argparse
import json
import logging
import sys
import copy
from pathlib import Path
from typing import Optional, Dict, Any
from scenario_engine import ScenarioExecutor, ScenarioError, ValidationError


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging based on verbosity level"""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def validate_scenario_file(path: str) -> Optional[Dict[str, Any]]:
    """Validate scenario file structure"""
    try:
        with open(path, 'r') as f:
            scenario = json.load(f)
        
        # Check required sections
        required = ['agents', 'actions', 'workflow']
        missing = [s for s in required if s not in scenario]
        
        if missing:
            print(f"Error: Scenario missing required sections: {', '.join(missing)}", file=sys.stderr)
            return None
        
        # Check for at least one agent
        if not scenario['agents']:
            print("Error: No agents defined", file=sys.stderr)
            return None
        
        # Check for at least one workflow step
        if not scenario['workflow']:
            print("Error: No workflow steps defined", file=sys.stderr)
            return None
        
        # Validate agent references and action references
        defined_agents = set(scenario['agents'].keys())
        defined_actions = set(scenario['actions'].keys())
        
        for i, step in enumerate(scenario['workflow']):
            step_id = step.get('id', f'step_{i}')
            
            # Check agent exists
            if 'agent' in step:
                agent_name = step['agent']
                if agent_name not in defined_agents:
                    print(f"Error: Step '{step_id}': Unknown agent '{agent_name}'", file=sys.stderr)
                    return None
            
            # Check action exists
            action_name = step.get('action')
            if action_name and action_name not in ['loop', 'clear_context', 'run_python', 'user_input', 'apply_patch']:
                if action_name not in defined_actions:
                    print(f"Error: Step '{step_id}': Unknown action '{action_name}'", file=sys.stderr)
                    return None
        
        return scenario
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in scenario file: {e}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"Error: Scenario file not found: {path}", file=sys.stderr)
        return None


def print_scenario_info(scenario: Dict[str, Any]) -> None:
    """Print information about a scenario"""
    metadata = scenario.get('metadata', {})
    print(f"\nScenario: {metadata.get('name', 'Unnamed')}")
    print(f"Version: {metadata.get('version', 'N/A')}")
    if 'description' in metadata:
        print(f"Description: {metadata['description']}")
    
    print(f"\nAgents: {len(scenario['agents'])}")
    for name, agent in scenario['agents'].items():
        temp = agent.get('temperature', 0.7)
        max_tokens = agent.get('maxContextTokens', 8000)
        timeout = agent.get('queryTimeout', 300)
        print(f"  - {name}: {agent['model']} (temp: {temp}, context: {max_tokens} tokens, timeout: {timeout}s)")
    
    print(f"\nActions: {len(scenario['actions'])}")
    for name, action in scenario['actions'].items():
        # Count input placeholders
        import re
        placeholders = re.findall(r'\{\{(\d+)\}\}', action['prompt'])
        input_count = len(set(placeholders))
        print(f"  - {name} ({input_count} inputs)")
    
    print(f"\nWorkflow Steps: {len(scenario['workflow'])}")
    
    # Configuration is now in metadata
    print(f"\nConfiguration:")
    print(f"  - Output Directory: {metadata.get('outputDirectory', './results')}")
    print(f"  - Log Level: {metadata.get('logLevel', 'INFO')}")
    print()


def create_example_scenario(output_path: str) -> None:
    """Create an example scenario file"""
    example = {
        "metadata": {
            "name": "Code Review Example",
            "version": "2.0",
            "description": "A simple example showing iterative code development with review",
            "outputDirectory": "./results",
            "logLevel": "info"
        },
        "agents": {
            "developer": {
                "model": "gemma:2b",
                "temperature": 0.7,
                "personality": "You are a helpful developer who writes clean Python code.",
                "maxContextTokens": 8000,
                "queryTimeout": 300
            },
            "reviewer": {
                "model": "gemma:2b",
                "temperature": 0.3,
                "personality": "You are a code reviewer who provides constructive feedback.",
                "maxContextTokens": 12000,
                "queryTimeout": 400
            }
        },
        "actions": {
            "writeCode": {
                "prompt": "Write a Python function that {{1}}"
            },
            "reviewCode": {
                "prompt": "Review this code and suggest improvements:\n\n{{1}}"
            },
            "improveCode": {
                "prompt": "Improve this code based on the feedback:\n\nOriginal code:\n{{1}}\n\nFeedback:\n{{2}}"
            },
            "createFinalReport": {
                "prompt": "Create a summary report:\n\n# Code Development Summary\n\n## Initial Implementation\n{{1}}\n\n## Review Feedback\n{{2}}\n\n## Final Implementation\n{{3}}\n\nAdd your analysis of the improvements made."
            }
        },
        "workflow": [
            {
                "action": "writeCode",
                "agent": "developer",
                "inputs": ["calculates the factorial of a number"],
                "output": "initial_code.md"
            },
            {
                "action": "reviewCode",
                "agent": "reviewer",
                "inputs": ["initial_code.md"],
                "output": "review_feedback.md"
            },
            {
                "action": "improveCode",
                "agent": "developer",
                "inputs": ["initial_code.md", "review_feedback.md"],
                "output": "improved_code.md"
            },
            {
                "action": "createFinalReport",
                "agent": "reviewer",
                "inputs": ["initial_code.md", "review_feedback.md", "improved_code.md"],
                "output": "development_report.md"
            },
            {
                "action": "improveCode",
                "agent": "developer",
                "inputs": ["improved_code.md", "Focus on extracting just the code"],
                "output": "factorial.py",
                "format": "python"
            }
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(example, f, indent=2)
    
    print(f"Created example scenario: {output_path}")


def update_output_directory(scenario: Dict[str, Any], iteration: int) -> Dict[str, Any]:
    """Update the output directory for the current iteration"""
    scenario_copy = copy.deepcopy(scenario)
    
    # Output directory is now in metadata
    if 'metadata' in scenario_copy and 'outputDirectory' in scenario_copy['metadata']:
        original_dir = scenario_copy['metadata']['outputDirectory']
        # Remove trailing slash if present
        original_dir = original_dir.rstrip('/')
        scenario_copy['metadata']['outputDirectory'] = f"{original_dir}_{iteration}"
    
    return scenario_copy


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description='Simplified Scenario-based Ollama Agent Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s scenario.json                    # Run once
  %(prog)s scenario.json 3                  # Run 3 times
  %(prog)s scenario.json --verbose          # Verbose output
  %(prog)s --validate scenario.json         # Validate only
  %(prog)s --info scenario.json             # Show scenario info
  %(prog)s --create-example my_scenario.json # Create example
  %(prog)s --dry-run scenario.json          # Show execution plan
        """
    )
    
    parser.add_argument('scenario', nargs='?',
                       help='Path to scenario JSON file')
    
    parser.add_argument('repetitions', nargs='?', type=int, default=1,
                       help='Number of times to run the scenario (default: 1)')
    
    parser.add_argument('--validate', '-c', action='store_true',
                       help='Validate scenario file without executing')
    
    parser.add_argument('--info', '-i', action='store_true',
                       help='Show scenario information without executing')
    
    parser.add_argument('--create-example', metavar='PATH',
                       help='Create an example scenario file')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress info logging (warnings and errors only)')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show execution plan without running')
    
    return parser


async def execute_single_iteration(scenario_path: str, scenario: Dict[str, Any], iteration: int, total_iterations: int, logger) -> bool:
    """Execute a single iteration of the scenario. Returns True if successful."""
    try:
        logger.info(f"Starting iteration {iteration}/{total_iterations}")
        
        # Update scenario with iteration-specific output directory
        iteration_scenario = update_output_directory(scenario, iteration)
        
        # Create a temporary scenario file for this iteration
        temp_scenario_path = f"{scenario_path}.temp_iter_{iteration}"
        
        try:
            with open(temp_scenario_path, 'w') as f:
                json.dump(iteration_scenario, f, indent=2)
            
            # Execute the scenario
            executor = ScenarioExecutor(temp_scenario_path)
            await executor.execute()
            
            logger.info(f"✓ Iteration {iteration}/{total_iterations} completed successfully")
            
            # Show output location
            metadata = iteration_scenario.get('metadata', {})
            output_dir = metadata.get('outputDirectory', './results')
            logger.info(f"  Outputs saved to: {output_dir}")
            
            return True
            
        finally:
            # Clean up temporary file
            try:
                Path(temp_scenario_path).unlink()
            except FileNotFoundError:
                pass
            
    except (ScenarioError, ValidationError) as e:
        logger.error(f"Iteration {iteration}/{total_iterations} failed - {type(e).__name__}: {e}")
        return False
        
    except KeyboardInterrupt:
        logger.warning(f"Iteration {iteration}/{total_iterations} cancelled by user")
        raise  # Re-raise to stop all iterations
        
    except Exception as e:
        logger.exception(f"Iteration {iteration}/{total_iterations} failed - Unexpected error: {e}")
        return False


async def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle example creation
    if args.create_example:
        create_example_scenario(args.create_example)
        return
    
    # Require scenario file for other operations
    if not args.scenario:
        parser.print_help()
        sys.exit(1)
    
    # Validate repetitions argument
    if args.repetitions < 1:
        print("Error: Number of repetitions must be at least 1", file=sys.stderr)
        sys.exit(1)
    
    # Setup logging
    setup_logging(args.verbose, args.quiet)
    logger = logging.getLogger(__name__)
    
    # Validate scenario
    scenario = validate_scenario_file(args.scenario)
    if not scenario:
        sys.exit(1)
    
    # Handle info display
    if args.info:
        print_scenario_info(scenario)
        if args.repetitions > 1:
            print(f"Note: Would run {args.repetitions} iterations with separate output directories")
        return
    
    # Handle validation only
    if args.validate:
        print(f"✓ Scenario file is valid: {args.scenario}")
        if args.repetitions > 1:
            print(f"Will run {args.repetitions} iterations when executed")
        return
    
    # Handle dry run
    if args.dry_run:
        print(f"Dry run of scenario: {args.scenario}")
        if args.repetitions > 1:
            print(f"Number of iterations: {args.repetitions}")
        print_scenario_info(scenario)
        print("Workflow steps:")
        for i, step in enumerate(scenario['workflow'], 1):
            step_id = step.get('id', f'step_{i}')
            action = step['action']
            if 'agent' in step:
                print(f"  {i}. [{step_id}] {step['agent']} → {action}")
                if 'inputs' in step:
                    print(f"      Inputs: {step['inputs']}")
                if 'output' in step:
                    print(f"      Output: {step['output']}")
            else:
                print(f"  {i}. [{step_id}] {action}")
                if action == 'loop':
                    print(f"      Iterations: {step.get('iterations', 1)}")
                elif action == 'run_python':
                    print(f"      Python execution with inputs: {step.get('inputs', [])}")
        
        if args.repetitions > 1:
            base_output_dir = scenario.get('metadata', {}).get('outputDirectory', './results')
            print(f"\nOutput directories that will be created:")
            for i in range(1, args.repetitions + 1):
                print(f"  {base_output_dir}_{i}")
        return
    
    # Execute scenario iterations
    try:
        if args.repetitions == 1:
            logger.info(f"Executing scenario: {args.scenario}")
        else:
            logger.info(f"Executing scenario: {args.scenario} ({args.repetitions} iterations)")
        
        successful_iterations = 0
        failed_iterations = 0
        
        for iteration in range(1, args.repetitions + 1):
            success = await execute_single_iteration(
                args.scenario, scenario, iteration, args.repetitions, logger
            )
            
            if success:
                successful_iterations += 1
            else:
                failed_iterations += 1
            
            # Add a small delay between iterations to ensure clean separation
            if iteration < args.repetitions:
                await asyncio.sleep(1)
        
        # Summary
        if args.repetitions == 1:
            if successful_iterations == 1:
                print("\n✓ Scenario execution completed successfully")
            else:
                print("\n✗ Scenario execution failed")
                sys.exit(1)
        else:
            print(f"\n=== Execution Summary ===")
            print(f"Total iterations: {args.repetitions}")
            print(f"Successful: {successful_iterations}")
            print(f"Failed: {failed_iterations}")
            
            if successful_iterations == args.repetitions:
                print("✓ All iterations completed successfully")
            elif successful_iterations > 0:
                print(f"⚠ {successful_iterations} of {args.repetitions} iterations completed successfully")
                sys.exit(1)
            else:
                print("✗ All iterations failed")
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
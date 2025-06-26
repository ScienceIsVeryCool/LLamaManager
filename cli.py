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
import jsonschema


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


def load_schema() -> Optional[Dict[str, Any]]:
    """Load the JSON schema for validation"""
    schema_path = Path(__file__).parent / 'schema.json'
    if not schema_path.exists():
        # If schema.json is not in the same directory, try current directory
        schema_path = Path('schema.json')
    
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: schema.json not found. Please ensure it's in the same directory as cli.py", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in schema file: {e}", file=sys.stderr)
        return None


def validate_scenario_file(path: str) -> Optional[Dict[str, Any]]:
    """Validate scenario file structure using jsonschema without modifying it."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            scenario = json.load(f)
        
        schema = load_schema()
        if not schema:
            return None
            
        try:
            jsonschema.validate(instance=scenario, schema=schema)
        except jsonschema.ValidationError as e:
            print(f"Error: Scenario validation failed: {e.message}", file=sys.stderr)
            return None
        
        # Additional validation: check for duplicate agent names
        agent_names = [agent['name'] for agent in scenario.get('agents', [])]
        if len(agent_names) != len(set(agent_names)):
            duplicates = {name for name in agent_names if agent_names.count(name) > 1}
            print(f"Error: Duplicate agent names found: {', '.join(duplicates)}", file=sys.stderr)
            return None
        
        # Additional validation: check for duplicate action names
        action_names = [action['name'] for action in scenario.get('actions', [])]
        if len(action_names) != len(set(action_names)):
            duplicates = {name for name in action_names if action_names.count(name) > 1}
            print(f"Error: Duplicate action names found: {', '.join(duplicates)}", file=sys.stderr)
            return None
        
        # --- MODIFICATION ---
        # The function no longer converts lists to dictionaries.
        # It returns the original, validated scenario structure.
        return scenario
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in scenario file: {e}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"Error: Scenario file not found: {path}", file=sys.stderr)
        return None


def print_scenario_info(scenario: Dict[str, Any]) -> None:
    """Print information about a scenario, iterating over lists."""
    config = scenario.get('config', {})
    print(f"\nScenario: {config.get('name', 'Unnamed')}")
    print(f"Version: {config.get('version', 'N/A')}")
    if 'description' in config:
        print(f"Description: {config['description']}")
    
    # --- MODIFICATION: Iterate over list of agents ---
    agents_list = scenario.get('agents', [])
    print(f"\nAgents: {len(agents_list)}")
    for agent in agents_list:
        name = agent['name']
        model = agent['model']
        temp = agent.get('temperature', 0.7)
        max_tokens = agent.get('maxContextTokens', 8000)
        timeout = agent.get('queryTimeout', 300)
        print(f"  - {name}: {model} (temp: {temp}, context: {max_tokens} tokens, timeout: {timeout}s)")
    
    # --- MODIFICATION: Iterate over list of actions ---
    actions_list = scenario.get('actions', [])
    print(f"\nActions: {len(actions_list)}")
    for action in actions_list:
        name = action['name']
        import re
        placeholders = re.findall(r'\{\{(\d+)\}\}', action['prompt'])
        input_count = len(set(placeholders))
        print(f"  - {name} ({input_count} inputs)")
    
    print(f"\nWorkflow Steps: {len(scenario.get('workflow', []))}")
    
    # Configuration
    print(f"\nConfiguration:")
    print(f"  - Work Directory: {config.get('workDir', './results')}")
    print(f"  - Python Timeout: {config.get('pythonTimeout', 30)}s")
    print(f"  - Log Level: {config.get('logLevel', 'INFO')}")
    print()

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


async def execute_single_iteration(scenario_path: str, loaded_scenario: Dict[str, Any], iteration: int, total_iterations: int, logger) -> bool:
    """Execute a single iteration of the scenario. Returns True if successful."""
    try:
        logger.info(f"Starting iteration {iteration}/{total_iterations}")
        
        try:
            # Pass the original, validated scenario to the executor
            executor = ScenarioExecutor(scenario_path, loaded_scenario)
            await executor.execute()
            
            logger.info(f"✓ Iteration {iteration}/{total_iterations} completed successfully")
            
            config = loaded_scenario.get('config', {})
            work_dir = config.get('workDir', './results')
            logger.info(f"  Outputs saved to: {work_dir}")
            
            return True
            
        finally:
            pass
            
    except (ScenarioError, ValidationError) as e:
        logger.error(f"Iteration {iteration}/{total_iterations} failed - {type(e).__name__}: {e}")
        return False
        
    except KeyboardInterrupt:
        logger.warning(f"Iteration {iteration}/{total_iterations} cancelled by user")
        raise
        
    except Exception as e:
        logger.exception(f"Iteration {iteration}/{total_iterations} failed - Unexpected error: {e}")
        return False


async def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
        
    if not args.scenario and not args.create_example:
        parser.print_help()
        sys.exit(1)

    # Note: create_example handling would go here if implemented

    if args.repetitions < 1:
        print("Error: Number of repetitions must be at least 1", file=sys.stderr)
        sys.exit(1)
    
    setup_logging(args.verbose, args.quiet)
    logger = logging.getLogger(__name__)
    
    scenario = validate_scenario_file(args.scenario)
    if not scenario:
        sys.exit(1)
    
    if args.info:
        print_scenario_info(scenario)
        if args.repetitions > 1:
            print(f"Note: Would run {args.repetitions} iterations.")
        return
    
    if args.validate:
        print(f"✓ Scenario file is valid: {args.scenario}")
        if args.repetitions > 1:
            print(f"Will run {args.repetitions} iterations when executed")
        return
    
    if args.dry_run:
        print(f"Dry run of scenario: {args.scenario}")
        if args.repetitions > 1:
            print(f"Number of iterations: {args.repetitions}")
        print_scenario_info(scenario)
        print("Workflow steps:")
        for i, step in enumerate(scenario.get('workflow', []), 1):
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
        return
    
    try:
        if args.repetitions == 1:
            logger.info(f"Executing scenario: {args.scenario}")
        else:
            logger.info(f"Executing scenario: {args.scenario} ({args.repetitions} iterations)")
        
        successful_iterations = 0
        failed_iterations = 0
        
        for iteration in range(1, args.repetitions + 1):
            # Pass the original 'scenario' dictionary
            success = await execute_single_iteration(
                args.scenario, scenario, iteration, args.repetitions, logger
            )
            
            if success:
                successful_iterations += 1
            else:
                failed_iterations += 1
            
            # --- MODIFICATION: Removed unnecessary sleep ---
        
        if args.repetitions > 1:
            print(f"\n=== Execution Summary ===")
            print(f"Total iterations: {args.repetitions}")
            print(f"Successful: {successful_iterations}")
            print(f"Failed: {failed_iterations}")
        
        if failed_iterations > 0:
            print(f"\n✗ Scenario execution failed on {failed_iterations} iteration(s).")
            sys.exit(1)
        else:
            print("\n✓ Scenario execution completed successfully.")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(130)
        
    except Exception as e:
        logger.exception("Unexpected error occurred")
        print(f"Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

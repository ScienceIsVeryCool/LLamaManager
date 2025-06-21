#!/usr/bin/env python3
"""
CLI for Scenario-based Ollama Agent Manager
"""

import asyncio
import argparse
import json
import logging
import sys
import copy
from pathlib import Path
from typing import Optional, Dict, Any
from scenario_engine import ScenarioExecutor, ScenarioError


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
        required = ['agentTemplates', 'actionTemplates', 'execution']
        missing = [s for s in required if s not in scenario]
        
        if missing:
            print(f"Error: Scenario missing required sections: {', '.join(missing)}", file=sys.stderr)
            return None
        
        # Check for at least one agent template
        if not scenario['agentTemplates']:
            print("Error: No agent templates defined", file=sys.stderr)
            return None
        
        # Check for at least one execution step
        if not scenario['execution']:
            print("Error: No execution steps defined", file=sys.stderr)
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
    info = scenario.get('scenario', {})
    print(f"\nScenario: {info.get('name', 'Unnamed')}")
    print(f"Version: {info.get('version', 'N/A')}")
    if 'description' in info:
        print(f"Description: {info['description']}")
    
    print(f"\nAgent Templates: {len(scenario['agentTemplates'])}")
    for name, template in scenario['agentTemplates'].items():
        print(f"  - {name}: {template['model']}")
    
    print(f"\nAction Templates: {len(scenario['actionTemplates'])}")
    for name in scenario['actionTemplates']:
        print(f"  - {name}")
    
    print(f"\nExecution Steps: {len(scenario['execution'])}")
    
    config = scenario.get('config', {})
    if config:
        print(f"\nConfiguration:")
        print(f"  - Output Directory: {config.get('outputDirectory', './results')}")
        print(f"  - Log Level: {config.get('logLevel', 'INFO')}")
        print(f"  - Query Timeout: {config.get('queryTimeout', 300)}s")
        print(f"  - Save Intermediate: {config.get('saveIntermediateOutputs', False)}")
    print()


def create_example_scenario(output_path: str) -> None:
    """Create an example scenario file"""
    example = {
        "scenario": {
            "name": "Code Review Example",
            "version": "1.0",
            "description": "A simple example showing iterative code development with review"
        },
        "agentTemplates": {
            "developer": {
                "model": "gemma:2b",
                "temperature": 0.7,
                "systemPrompt": "You are a helpful developer who writes clean Python code."
            },
            "reviewer": {
                "model": "gemma:2b",
                "temperature": 0.3,
                "systemPrompt": "You are a code reviewer who provides constructive feedback.",
                "defaultContext": "clean"
            }
        },
        "actionTemplates": {
            "writeCode": {
                "promptTemplate": "Write a Python function that {{task}}"
            },
            "reviewCode": {
                "promptTemplate": "Review this code and suggest improvements:\n\n{{code}}"
            },
            "improveCode": {
                "promptTemplate": "Improve this code based on the feedback:\n\nOriginal code:\n{{code}}\n\nFeedback:\n{{feedback}}"
            }
        },
        "execution": [
            {
                "action": "createAgent",
                "params": {
                    "template": "developer",
                    "instanceName": "dev1"
                }
            },
            {
                "action": "createAgent",
                "params": {
                    "template": "reviewer",
                    "instanceName": "reviewer1"
                }
            },
            {
                "action": "writeCode",
                "agent": "dev1",
                "params": {
                    "task": "calculates the factorial of a number"
                },
                "output": "initial_code"
            },
            {
                "action": "reviewCode",
                "agent": "reviewer1",
                "params": {
                    "code": "{{outputs.initial_code}}"
                },
                "output": "review_feedback"
            },
            {
                "action": "improveCode",
                "agent": "dev1",
                "params": {
                    "code": "{{outputs.initial_code}}",
                    "feedback": "{{outputs.review_feedback}}"
                },
                "output": "final_code"
            },
            {
                "action": "saveToFile",
                "params": {
                    "content": "# Final Implementation\n\n{{outputs.final_code}}",
                    "filename": "factorial.py"
                }
            }
        ],
        "config": {
            "logLevel": "info",
            "saveIntermediateOutputs": true,
            "outputDirectory": "./results",
            "queryTimeout": 300
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(example, f, indent=2)
    
    print(f"Created example scenario: {output_path}")


def update_output_directory(scenario: Dict[str, Any], iteration: int) -> Dict[str, Any]:
    """Update the output directory for the current iteration"""
    scenario_copy = copy.deepcopy(scenario)
    
    if 'config' in scenario_copy and 'outputDirectory' in scenario_copy['config']:
        original_dir = scenario_copy['config']['outputDirectory']
        # Remove trailing slash if present
        original_dir = original_dir.rstrip('/')
        scenario_copy['config']['outputDirectory'] = f"{original_dir}_{iteration}"
    
    return scenario_copy


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description='Scenario-based Ollama Agent Manager',
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
            config = iteration_scenario.get('config', {})
            if config.get('saveIntermediateOutputs'):
                output_dir = config.get('outputDirectory', './results')
                logger.info(f"  Outputs saved to: {output_dir}")
            
            return True
            
        finally:
            # Clean up temporary file
            try:
                Path(temp_scenario_path).unlink()
            except FileNotFoundError:
                pass
            
    except ScenarioError as e:
        logger.error(f"Iteration {iteration}/{total_iterations} failed - Scenario error: {e}")
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
        print("Execution steps:")
        for i, step in enumerate(scenario['execution'], 1):
            step_id = step.get('id', f'step_{i}')
            action = step['action']
            if 'agent' in step:
                print(f"  {i}. [{step_id}] {step['agent']} → {action}")
            else:
                print(f"  {i}. [{step_id}] {action}")
        
        if args.repetitions > 1:
            base_output_dir = scenario.get('config', {}).get('outputDirectory', './results')
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
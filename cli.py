#!/usr/bin/env python3
"""
CLI for Scenario-based Ollama Agent Manager
"""

import asyncio
import argparse
import json
import logging
import sys
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
    
    print(f"\nAgent Templates: {len(scenario['agentTemplates'])}")
    for name, template in scenario['agentTemplates'].items():
        print(f"  - {name}: {template['model']}")
    
    print(f"\nAction Templates: {len(scenario['actionTemplates'])}")
    for name in scenario['actionTemplates']:
        print(f"  - {name}")
    
    print(f"\nExecution Steps: {len(scenario['execution'])}")
    print()


def create_example_scenario(output_path: str) -> None:
    """Create an example scenario file"""
    example = {
        "scenario": {
            "name": "Code Review Example",
            "version": "1.0"
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
                "systemPrompt": "You are a code reviewer who provides constructive feedback."
            }
        },
        "actionTemplates": {
            "writeCode": {
                "type": "prompt",
                "promptTemplate": "Write a Python function that {{task}}",
                "outputCapture": "full"
            },
            "reviewCode": {
                "type": "prompt",
                "promptTemplate": "Review this code and suggest improvements:\n\n{{code}}",
                "inputRequired": ["code"],
                "outputCapture": "full"
            }
        },
        "execution": [
            {
                "id": "create_dev",
                "action": "createAgent",
                "params": {
                    "template": "developer",
                    "instanceName": "dev1"
                }
            },
            {
                "id": "create_reviewer",
                "action": "createAgent",
                "params": {
                    "template": "reviewer",
                    "instanceName": "reviewer1"
                }
            },
            {
                "id": "write_function",
                "action": "writeCode",
                "agent": "dev1",
                "params": {
                    "task": "calculates the factorial of a number"
                },
                "output": "initial_code"
            },
            {
                "id": "review_function",
                "action": "reviewCode",
                "agent": "reviewer1",
                "params": {
                    "code": "{{outputs.initial_code}}"
                }
            }
        ],
        "config": {
            "logLevel": "info",
            "saveIntermediateOutputs": True,
            "outputDirectory": "./results"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(example, f, indent=2)
    
    print(f"Created example scenario: {output_path}")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description='Scenario-based Ollama Agent Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s scenario.json
  %(prog)s scenario.json --verbose
  %(prog)s --validate scenario.json
  %(prog)s --create-example my_scenario.json
        """
    )
    
    parser.add_argument('scenario', nargs='?',
                       help='Path to scenario JSON file')
    
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
        return
    
    # Handle validation only
    if args.validate:
        print(f"✓ Scenario file is valid: {args.scenario}")
        return
    
    # Handle dry run
    if args.dry_run:
        print(f"Dry run of scenario: {args.scenario}")
        print_scenario_info(scenario)
        print("Execution steps:")
        for i, step in enumerate(scenario['execution'], 1):
            print(f"  {i}. {step.get('id', 'unnamed')} - Action: {step['action']}")
        return
    
    # Execute scenario
    try:
        logger.info(f"Executing scenario: {args.scenario}")
        
        executor = ScenarioExecutor(args.scenario)
        await executor.execute()
        
        print("\n✓ Scenario execution completed successfully")
        
        # Show output location if configured
        config = scenario.get('config', {})
        if config.get('saveIntermediateOutputs'):
            output_dir = config.get('outputDirectory', './results')
            print(f"  Outputs saved to: {output_dir}")
        
    except ScenarioError as e:
        logger.error(f"Scenario error: {e}")
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
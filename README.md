# Scenario-based Ollama Agent System

A modular, scenario-driven system for orchestrating multi-agent conversations with Ollama models.

## Key Features

- **Scenario-driven execution**: Define complex agent interactions in JSON
- **Memory-efficient**: Only one model loaded at a time
- **Template system**: Reusable agent and action templates
- **Flexible data flow**: Pass outputs between agents using template variables
- **Loop support**: Iterate actions with variable substitution
- **Context management**: Control how agents maintain conversation history

## Quick Start

1. Ensure Ollama is running (`ollama serve`)
2. Create or use an example scenario file
3. Run the scenario:

```bash
python cli.py cellular_automata_scenario.json
```

## Architecture

### Core Components

1. **ScenarioExecutor**: Main execution engine that processes scenario files
2. **AgentInstance**: Runtime representation of an agent with conversation management
3. **TemplateEngine**: Handles variable substitution in prompts and parameters
4. **Action System**: Extensible action execution framework

### Memory Management

- Models are loaded on-demand when an agent needs to execute
- Previous models are unloaded before loading new ones
- Agents maintain their conversation history independently

### Scenario File Structure

```json
{
  "scenario": {
    "name": "Scenario Name",
    "version": "1.0"
  },
  
  "agentTemplates": {
    "templateName": {
      "model": "model_name",
      "temperature": 0.7,
      "systemPrompt": "System prompt",
      "defaultContext": "clean"  // clean | append | rolling
    }
  },
  
  "actionTemplates": {
    "actionName": {
      "type": "prompt",
      "promptTemplate": "Template with {{variables}}",
      "inputRequired": ["variable_names"],
      "outputCapture": "full"
    }
  },
  
  "execution": [
    // Execution steps
  ],
  
  "config": {
    "logLevel": "info",
    "saveIntermediateOutputs": true,
    "outputDirectory": "./results"
  }
}
```

## Built-in Actions

### createAgent
Creates an agent instance from a template:
```json
{
  "action": "createAgent",
  "params": {
    "template": "templateName",
    "instanceName": "agent1"
  }
}
```

### loop
Executes steps multiple times with iteration variable:
```json
{
  "action": "loop",
  "iterations": 3,
  "steps": [
    // Steps to repeat
  ]
}
```

### saveToFile
Saves content to a file:
```json
{
  "action": "saveToFile",
  "params": {
    "content": "{{variable}}",
    "filename": "output.txt"
  }
}
```

## Template Variables

### Basic Variables
- `{{outputs.stepId}}` - Access output from a specific step
- `{{variable}}` - Access a parameter variable

### Functions
- `{{lastOutput('agentName')}}` - Get the last output from an agent

### Loop Variables
- `{{iteration}}` - Current iteration number in a loop

## Context Modes

- **clean**: Each query starts fresh (default)
- **append**: Full conversation history is maintained
- **rolling**: Keep last 10 messages for context

## CLI Usage

```bash
# Run a scenario
python cli.py scenario.json

# Validate without executing
python cli.py --validate scenario.json

# Show scenario information
python cli.py --info scenario.json

# Create example scenario
python cli.py --create-example my_scenario.json

# Verbose logging
python cli.py scenario.json --verbose

# Dry run (show execution plan)
python cli.py scenario.json --dry-run
```

## Example: Code Development Workflow

The included `cellular_automata_scenario.json` demonstrates:
1. Creating developer and reviewer agents
2. Generating initial code solution
3. Review and feedback loop
4. Multiple revision iterations
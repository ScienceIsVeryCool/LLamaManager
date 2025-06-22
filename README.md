# Scenario-based Ollama Agent System

A clean, scenario-driven system for orchestrating multi-agent conversations with Ollama models.

## Key Features

- **JSON-based scenarios**: Define complex agent interactions in simple JSON
- **Memory-efficient**: Only one model loaded at a time  
- **Template system**: Reusable agent and action templates with automatic parameter validation
- **Flexible data flow**: Pass outputs between agents using template variables
- **Loop support**: Iterate actions with variable substitution
- **Context management**: Control how agents maintain conversation history
- **Automatic context window management**: Prevents context overflow by trimming old messages

## Quick Start

1. Ensure Ollama is running (`ollama serve`)
2. Create a scenario file (see examples below)
3. Run the scenario:

```bash
python cli.py my_scenario.json
```

## Scenario Structure

```json
{
  "scenario": {
    "name": "My Scenario",
    "version": "1.0"
  },
  
  "agentTemplates": {
    "developer": {
      "model": "gemma:2b",
      "temperature": 0.7,
      "systemPrompt": "You are a helpful developer.",
      "defaultContext": "clean"  // clean | append | rolling
    }
  },
  
  "actionTemplates": {
    "writeCode": {
      "promptTemplate": "Write {{language}} code that {{task}}"
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
      "action": "writeCode",
      "agent": "dev1",
      "params": {
        "language": "Python",
        "task": "sorts a list"
      },
      "output": "sorting_code"  // Optional: store output for later use
    }
  ],
  
  "config": {
    "logLevel": "info",
    "outputDirectory": "./results",
    "saveIntermediateOutputs": false,
    "queryTimeout": 300,
    "maxContextTokens": 8000  // Optional: max tokens before trimming context
  }
}
```

## Core Concepts

### Agents
Agents are instances created from templates. Each agent maintains its own conversation history and can use different models.

### Actions
Actions are reusable prompts with variable substitution. The system automatically validates that all template variables have corresponding parameters.

### Outputs
Any action can store its output by adding an `output` field. These outputs can be referenced later using `{{outputs.name}}`.

### Context Modes
- **clean**: Each query starts fresh (default)
- **append**: Full conversation history is maintained
- **rolling**: Keep last 10 messages for context

## Built-in Actions

### createAgent
Creates an agent instance:
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
Repeats steps with an iteration counter:
```json
{
  "action": "loop",
  "iterations": 3,
  "steps": [
    {
      "id": "step_{{iteration}}",  // {{iteration}} is replaced with 1, 2, 3
      "action": "someAction",
      "agent": "agent1",
      "params": {}
    }
  ]
}
```

### saveToFile
Saves content to a file:
```json
{
  "action": "saveToFile",
  "params": {
    "content": "{{outputs.someOutput}}",
    "filename": "output.txt"
  }
}
```

### clearContext
Clears conversation history for an agent:
```json
{
  "action": "clearContext",
  "agent": "agent1"
}
```

## Template Variables

### Basic Substitution
- `{{paramName}}` - Replaced with parameter value
- `{{outputs.outputName}}` - Access stored output
- `{{iteration}}` - Current loop iteration (inside loops)

### Functions
- `{{lastOutput('agentName')}}` - Get the last output from an agent

## CLI Usage

```bash
# Run a scenario
python cli.py scenario.json

# Run multiple times
python cli.py scenario.json 3

# Validate scenario
python cli.py --validate scenario.json

# Show scenario info
python cli.py --info scenario.json

# Create example scenario
python cli.py --create-example example.json

# Dry run (show execution plan)
python cli.py --dry-run scenario.json
```

## Complete Example

Here's a simple code review scenario:

```json
{
  "scenario": {
    "name": "Code Review",
    "version": "1.0"
  },
  
  "agentTemplates": {
    "developer": {
      "model": "gemma:2b",
      "temperature": 0.7,
      "systemPrompt": "You write clean Python code."
    },
    "reviewer": {
      "model": "gemma:2b", 
      "temperature": 0.3,
      "systemPrompt": "You provide constructive code reviews."
    }
  },
  
  "actionTemplates": {
    "implement": {
      "promptTemplate": "Implement a {{type}} that {{description}}"
    },
    "review": {
      "promptTemplate": "Review this code:\n\n{{code}}\n\nFocus on: {{focus}}"
    },
    "revise": {
      "promptTemplate": "Revise this code based on feedback:\n\nCode:\n{{code}}\n\nFeedback:\n{{feedback}}"
    }
  },
  
  "execution": [
    {
      "action": "createAgent",
      "params": {"template": "developer", "instanceName": "dev"}
    },
    {
      "action": "createAgent", 
      "params": {"template": "reviewer", "instanceName": "reviewer"}
    },
    {
      "action": "implement",
      "agent": "dev",
      "params": {
        "type": "function",
        "description": "calculates factorial"
      },
      "output": "initial_code"
    },
    {
      "action": "review",
      "agent": "reviewer",
      "params": {
        "code": "{{outputs.initial_code}}",
        "focus": "efficiency and correctness"
      },
      "output": "review_feedback"
    },
    {
      "action": "revise",
      "agent": "dev",
      "params": {
        "code": "{{outputs.initial_code}}",
        "feedback": "{{outputs.review_feedback}}"
      },
      "output": "final_code"
    },
    {
      "action": "saveToFile",
      "params": {
        "content": "{{outputs.final_code}}",
        "filename": "factorial.py"
      }
    }
  ],
  
  "config": {
    "outputDirectory": "./output"
  }
}
```

## Tips

1. **IDs are optional**: Only add an `id` field if you need it for debugging or filenames
2. **Parameter validation is automatic**: The system checks that all `{{variables}}` in templates have matching parameters
3. **Use descriptive output names**: Instead of `output1`, use names like `initial_code` or `review_feedback`
4. **Start simple**: Test with small scenarios before building complex workflows
5. **Use intermediate outputs**: Enable `saveIntermediateOutputs` in config for debugging
6. **Context window management**: The system automatically trims old messages if approaching token limits
7. **Clear context strategically**: Use `clearContext` action to reset agent memory when needed

## Troubleshooting

### Common Issues

1. **Nested template variables**: Avoid patterns like `{{outputs.name_{{iteration}}}}`. Instead, use functions like `{{lastOutput('agent')}}` or restructure your approach.

2. **Context overflow**: If agents are having issues with long conversations, try:
   - Setting `maxContextTokens` to a lower value in config
   - Using `clearContext` action between major phases
   - Setting `defaultContext: "rolling"` instead of `"append"`

3. **Memory issues with large models**: The system only loads one model at a time, but ensure Ollama has enough memory allocated.

4. **Slow execution**: 
   - Reduce `temperature` for more deterministic outputs
   - Set lower `queryTimeout` values to fail faster
   - Use smaller models for iterative tasks

## Advanced Examples

The repository includes three example scenarios:

1. **cellular_automata_scenario_simple.json**: Basic iterative development with one developer and reviewer
2. **cellular_automata_competitive.json**: Two developers with different personalities compete, with a senior reviewer providing feedback through multiple iterations  
3. **cellular_automata_pyramid.json**: Tournament-style development where four developers compete in rounds, with a senior developer improving the winning design and an architect providing final approval
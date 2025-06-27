# Scenario-based Ollama Agent System

A clean, human-readable system for orchestrating multi-agent conversations with Ollama models. Define complex agent interactions in simple JSON files and let the system handle the execution.

## Key Features

- **Simple JSON scenarios**: Easy-to-write, easy-to-understand configuration files
- **Schema validation**: Automatic validation against JSON schema ensures correctness
- **File-based workflow**: All inputs and outputs are files - no complex variable tracking
- **Multi-agent orchestration**: Coordinate multiple AI agents with different personalities
- **Python execution testing**: Run and test generated Python code automatically
- **Interactive user input**: Pause workflow for human input and review
- **Memory efficient**: Only one model loaded at a time
- **Automatic context management**: Agents automatically manage their conversation history
- **Built-in iteration support**: Loop through workflows with automatic variable substitution

## Quick Start

1. Ensure Ollama is running (`ollama serve`)
2. Install required Python packages:
   ```bash
   pip install jsonschema
   ```
3. Create a `testenv` virtual environment in your project directory:
   ```bash
   python -m venv testenv
   source testenv/bin/activate
   pip install matplotlib numpy  # Install required packages
   deactivate
   ```
4. Create a scenario file (see examples below)
5. Run the scenario:

```bash
python cli.py my_scenario.json
```

## Required Files

The system requires two essential files in the same directory:
- `schema.json`: Defines the structure and validation rules for scenario files
- `scenario_engine.py`: The main execution engine
- `cli.py`: Command-line interface

## Scenario Structure

A scenario file must conform to the JSON schema and has these main sections:

```json
{
  "config": {
    "name": "My Scenario",
    "version": "1.0",
    "description": "What this scenario does",
    "workDir": "./results",
    "logLevel": "info"
  },
  
  "agents": [
    {
      "name": "assistant",
      "model": "gemma:2b",
      "temperature": 0.7,
      "personality": "You are a helpful assistant.",
      "maxContextTokens": 8000,
      "queryTimeout": 300
    }
  ],
  
  "actions": [
    {
      "name": "analyze",
      "prompt": "Analyze this {{1}} and provide insights about {{2}}"
    }
  ],
  
  "workflow": [
    {
      "action": "analyze",
      "agent": "assistant",
      "inputs": ["data.txt", "performance patterns"],
      "output": "analysis.md"
    }
  ]
}
```

## Core Concepts

### Agents
Define AI agents with specific models and personalities. Agents are defined as an array, with each agent having a unique name:

```json
"agents": [
  {
    "name": "developer",
    "model": "gemma:2b",
    "temperature": 0.7,
    "personality": "You are an expert Python developer who writes clean, efficient code.",
    "maxContextTokens": 12000,
    "queryTimeout": 400
  },
  {
    "name": "reviewer",
    "model": "gemma:2b", 
    "temperature": 0.3,
    "personality": "You are a thorough code reviewer who provides constructive feedback.",
    "maxContextTokens": 8000,
    "queryTimeout": 300
  }
]
```

**Agent Configuration:**
- `name`: Unique identifier for the agent (required)
- `model`: The Ollama model to use (required)
- `temperature`: Creativity level (0.0 to 1.0, optional, default: 0.7)
- `personality`: System prompt defining the agent's role (optional)
- `maxContextTokens`: Maximum tokens before trimming conversation history (optional, default: 8000)
- `queryTimeout`: Maximum seconds to wait for responses (optional, default: 300)

**Context Management:**
All agents maintain their full conversation history until the context window approaches the limit, at which point older messages are automatically trimmed while preserving the system prompt and most recent interactions.

### Actions
Actions are reusable prompt templates with numbered placeholders. Actions are defined as an array, with each action having a unique name:

```json
"actions": [
  {
    "name": "writeCode",
    "prompt": "Write a {{1}} function that {{2}}. Use best practices and include docstrings."
  },
  {
    "name": "reviewCode",
    "prompt": "Review this {{1}} code:\n\n{{2}}\n\nFocus on: {{3}}"
  }
]
```

### Workflow
The workflow defines the sequence of actions to execute:

```json
"workflow": [
  {
    "action": "writeCode",
    "agent": "developer",
    "inputs": ["Python", "calculates fibonacci numbers"],
    "output": "fibonacci.py",
    "format": "python"  // Optional: extracts code from markdown
  },
  {
    "action": "reviewCode",
    "agent": "reviewer",
    "inputs": ["Python", "fibonacci.py", "efficiency and correctness"],
    "output": "review.md"
  }
]
```

### File I/O
The system automatically handles file reading and writing with a simple rule:
- **Reading**: If an input string contains **no spaces**, it's treated as a filename and read automatically. If it contains spaces, it's used as literal text.
- **Writing**: The `output` field specifies where to save the result
- **Format**: Use `"format": "python"` to extract code blocks from markdown responses
- **Work Directory**: All files are read from and written to the work directory specified in config

**Input Examples:**
```json
"inputs": ["data.txt", "analyze the performance trends"]
```
In this example, `data.txt` (no spaces) is read as a file, while `analyze the performance trends` (contains spaces) is used as literal text.

## Built-in Actions

### Loop
Execute steps multiple times with automatic variable substitution:

```json
{
  "action": "loop",
  "iterations": 3,
  "steps": [
    {
      "action": "generate",
      "agent": "creator",
      "inputs": ["Create variant {{iteration}} of the design"],
      "output": "design_v{{iteration}}.txt"
    }
  ]
}
```

### Clear Context
Reset an agent's conversation history:

```json
{
  "action": "clear_context",
  "agent": "reviewer"
}
```

### Run Python
Execute Python scripts in a separate virtual environment and capture output:

```json
{
  "action": "run_python",
  "inputs": ["script.py", "arg1", "arg2"],
  "output": "execution_results.log"
}
```

**Run Python Features:**
- Executes in a separate `testenv` virtual environment
- 30-second timeout (terminates early on errors)
- Captures both stdout and stderr
- Supports command-line arguments
- Returns execution status and output for agent analysis

**Setup Requirements:**
Create a `testenv` virtual environment in your project directory:
```bash
python -m venv testenv
source testenv/bin/activate
pip install matplotlib numpy pandas  # Install packages as needed
deactivate
```

### User Input
Pause execution and wait for user input from the terminal:

```json
{
  "action": "user_input",
  "inputs": ["summary.txt", "Please review the above summary"],
  "output": "user_feedback.txt"
}
```

**User Input Features:**
- Displays all input files and text to the user in the terminal
- Pauses workflow execution indefinitely until user provides input
- Saves user's response to the specified output file
- Supports both file inputs and literal text prompts
- Workflow continues normally after user input is collected

**User Input Behavior:**
1. The system displays a formatted header indicating user input is required
2. All inputs are shown to the user (files are read and displayed, text is shown as-is)
3. The user is prompted to enter a response
4. The response is saved to the output file
5. Execution continues with the next workflow step

**Example Use Cases:**
- Manual review and approval of generated content
- Collecting user requirements mid-workflow
- Human quality control checkpoints
- Interactive decision making in automated processes

## Schema Validation

All scenario files are automatically validated against the JSON schema before execution. The validation ensures:

1. **Required fields are present**: `agents` and `actions` arrays must exist
2. **Field types are correct**: All fields have the expected data types
3. **No duplicate names**: Agent names and action names must be unique within their respective arrays
4. **Workflow references are valid**: The system validates that all agent and action references in the workflow exist

If validation fails, you'll receive clear error messages indicating what needs to be fixed.

## Complete Example: Interactive Code Development

```json
{
  "config": {
    "name": "Interactive Code Development",
    "version": "1.0",
    "description": "Generate code with human review and approval",
    "workDir": "./interactive_output",
    "logLevel": "info"
  },
  
  "agents": [
    {
      "name": "developer",
      "model": "gemma:2b",
      "temperature": 0.7,
      "personality": "You are a Python developer who values clean, efficient code.",
      "maxContextTokens": 10000,
      "queryTimeout": 400
    },
    {
      "name": "improver",
      "model": "gemma:2b",
      "temperature": 0.5,
      "personality": "You are a senior developer who improves code based on feedback.",
      "maxContextTokens": 8000,
      "queryTimeout": 300
    }
  ],
  
  "actions": [
    {
      "name": "implement",
      "prompt": "Implement a {{1}} that {{2}}. Include proper error handling and documentation."
    },
    {
      "name": "improve",
      "prompt": "Improve this code based on the following feedback:\n\nOriginal Code:\n{{1}}\n\nUser Feedback:\n{{2}}\n\nProvide the improved version."
    }
  ],
  
  "workflow": [
    {
      "action": "implement",
      "agent": "developer",
      "inputs": ["Python script", "creates a data visualization dashboard"],
      "output": "dashboard.py",
      "format": "python"
    },
    {
      "action": "user_input",
      "inputs": ["dashboard.py", "Please review the generated code above. Provide feedback or suggestions for improvement:"],
      "output": "user_feedback.txt"
    },
    {
      "action": "improve",
      "agent": "improver",
      "inputs": ["dashboard.py", "user_feedback.txt"],
      "output": "dashboard_improved.py",
      "format": "python"
    },
    {
      "action": "run_python",
      "inputs": ["dashboard_improved.py"],
      "output": "execution_results.log"
    },
    {
      "action": "user_input",
      "inputs": ["execution_results.log", "Final approval - does the output look correct? (yes/no):"],
      "output": "final_approval.txt"
    }
  ]
}
```

## CLI Usage

```bash
# Run a scenario
python cli.py scenario.json

# Run multiple times (uses same workDir but keeps backups)
python cli.py scenario.json 3

# Validate scenario without executing
python cli.py --validate scenario.json

# Show scenario information
python cli.py --info scenario.json

# Create an example scenario
python cli.py --create-example example.json

# Dry run (show execution plan)
python cli.py --dry-run scenario.json

# Verbose output
python cli.py scenario.json --verbose

# Quiet mode (warnings and errors only)
python cli.py scenario.json --quiet
```

## Configuration Options

### Agent Options
- `name`: Unique identifier for the agent (required)
- `model`: The Ollama model to use (required)
- `temperature`: Creativity level (0.0 to 1.0, default: 0.7)
- `personality`: System prompt defining the agent's role
- `maxContextTokens`: Maximum tokens before trimming conversation history (default: 8000)
- `queryTimeout`: Maximum seconds to wait for model responses (default: 300)

### Action Options
- `name`: Unique identifier for the action (required)
- `prompt`: Template string with {{1}}, {{2}}, etc. placeholders (required)

### Config Section
- `name`: Scenario name for identification
- `version`: Scenario version
- `description`: Brief description of what the scenario does
- `workDir`: Directory where all files are saved and commands are executed (default: "./results")
- `logLevel`: Logging verbosity ("debug", "info", "warning", "error")

## Tips & Best Practices

1. **Schema Compliance**: Always ensure your scenario file validates against the schema
2. **Unique Names**: Agent and action names must be unique
3. **Input Rules**: Use inputs without spaces for filenames, inputs with spaces for literal text
4. **File Extensions**: Include extensions (.py, .md, .txt) for clarity in file outputs
5. **Action Design**: Keep actions focused on a single task for better reusability
6. **Context Management**: Use `clear_context` between major workflow phases to avoid confusion
7. **Temperature Settings**: Lower temperatures (0.3) for analytical tasks, higher (0.7+) for creative tasks
8. **Context Limits**: Adjust `maxContextTokens` based on your model's capabilities and task complexity
9. **Timeout Settings**: Set longer `queryTimeout` for complex reasoning tasks
10. **Work Directory**: Use descriptive work directory names to organize different scenario runs
11. **Python Testing**: Use `run_python` to validate generated code automatically
12. **Virtual Environment**: Keep `testenv` isolated with only necessary packages
13. **Interactive Workflows**: Use `user_input` for quality control and approval checkpoints
14. **User Experience**: Provide clear prompts and context when requesting user input

## Troubleshooting

### Schema Validation Errors
- Ensure your JSON is valid (use a JSON validator)
- Check that all required fields are present
- Verify that agent and action names are unique
- Make sure field types match the schema (arrays vs objects)
- Ensure `user_input` actions are included as valid action types in your schema

### File Not Found Errors
- Ensure the file was created in a previous step
- Check that you're using the correct filename (no spaces for file inputs)
- Verify the work directory path

### Input Processing Issues
- Remember: inputs with spaces = literal text, inputs without spaces = file names
- If you need a literal filename with spaces, use a symlink or rename the file
- Check work directory for file existence

### Context Window Errors
- Increase `maxContextTokens` in agent definition for complex conversations
- Add `clear_context` actions between major phases
- Consider breaking complex workflows into smaller steps

### Model Loading Issues
- Ensure the model is installed: `ollama pull model_name`
- Check that Ollama is running: `ollama serve`
- Verify model names match exactly (case-sensitive)

### Python Execution Issues
- Ensure `testenv` virtual environment exists in project directory
- Install required packages in testenv: `source testenv/bin/activate && pip install package_name`
- Check execution logs for specific error messages
- Verify Python script syntax before execution

### Agent Timeout Issues
- Increase `queryTimeout` in agent definition for complex reasoning tasks
- Monitor model performance and adjust timeouts accordingly
- Consider using simpler prompts if timeouts persist

### User Input Issues
- Ensure terminal input is available (not running in background processes)
- Use Ctrl+C to cancel user input if needed
- Check that output file is being created correctly
- Verify input files exist before user_input step

### Virtual Environment Setup
```bash
# Create testenv
python -m venv testenv

# Activate and install common packages
source testenv/bin/activate
pip install matplotlib numpy pandas requests beautifulsoup4
deactivate
```
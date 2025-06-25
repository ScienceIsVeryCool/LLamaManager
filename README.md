# Scenario-based Ollama Agent System

A clean, human-readable system for orchestrating multi-agent conversations with Ollama models. Define complex agent interactions in simple JSON files and let the system handle the execution.

## Key Features

- **Simple JSON scenarios**: Easy-to-write, easy-to-understand configuration files
- **File-based workflow**: All inputs and outputs are files - no complex variable tracking
- **Multi-agent orchestration**: Coordinate multiple AI agents with different personalities
- **Python execution testing**: Run and test generated Python code automatically
- **Memory efficient**: Only one model loaded at a time
- **Flexible context modes**: Control how agents remember conversations
- **Automatic validation**: Catches configuration errors before execution
- **Built-in iteration support**: Loop through workflows with automatic variable substitution

## Quick Start

1. Ensure Ollama is running (`ollama serve`)
2. Create a `testenv` virtual environment in your project directory:
   ```bash
   python -m venv testenv
   source testenv/bin/activate
   pip install matplotlib numpy  # Install required packages
   deactivate
   ```
3. Create a scenario file (see examples below)
4. Run the scenario:

```bash
python cli.py my_scenario.json
```

## Scenario Structure

A scenario file has four main sections:

```json
{
  "metadata": {
    "name": "My Scenario",
    "version": "1.0",
    "description": "What this scenario does"
  },
  
  "agents": {
    "assistant": {
      "model": "gemma:2b",
      "temperature": 0.7,
      "personality": "You are a helpful assistant.",
      "contextType": "clean"  // clean | append | rolling
    }
  },
  
  "actions": {
    "analyze": {
      "prompt": "Analyze this {{1}} and provide insights about {{2}}"
    }
  },
  
  "workflow": [
    {
      "action": "analyze",
      "agent": "assistant",
      "inputs": ["data.txt", "performance patterns"],
      "output": "analysis.md"
    }
  ],
  
  "config": {
    "logLevel": "info",
    "outputDirectory": "./results",
    "queryTimeout": 300,
    "maxContextTokens": 8000
  }
}
```

## Core Concepts

### Agents
Define AI agents with specific models and personalities:

```json
"agents": {
  "developer": {
    "model": "gemma:2b",
    "temperature": 0.7,
    "personality": "You are an expert Python developer who writes clean, efficient code.",
    "contextType": "rolling"
  },
  "reviewer": {
    "model": "gemma:2b", 
    "temperature": 0.3,
    "personality": "You are a thorough code reviewer who provides constructive feedback.",
    "contextType": "clean"
  }
}
```

**Context Types:**
- `clean`: Each query starts fresh
- `append`: Full conversation history is maintained
- `rolling`: Keep last 10 messages for context

### Actions
Actions are reusable prompt templates with numbered placeholders:

```json
"actions": {
  "writeCode": {
    "prompt": "Write a {{1}} function that {{2}}. Use best practices and include docstrings."
  },
  "reviewCode": {
    "prompt": "Review this {{1}} code:\n\n{{2}}\n\nFocus on: {{3}}"
  }
}
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
The system automatically handles file reading and writing:
- **Reading**: If an input looks like a filename (has extension or path separators), it's read automatically
- **Writing**: The `output` field specifies where to save the result
- **Format**: Use `"format": "python"` to extract code blocks from markdown responses

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
- 60-second timeout (terminates early on errors)
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

## Complete Example: Code Development with Testing

```json
{
  "metadata": {
    "name": "Code Development with Testing",
    "version": "1.0",
    "description": "Generate, execute, and analyze Python code"
  },
  
  "agents": {
    "developer": {
      "model": "gemma:2b",
      "temperature": 0.7,
      "personality": "You are a Python developer who values clean, efficient code.",
      "contextType": "append"
    },
    "tester": {
      "model": "gemma:2b",
      "temperature": 0.3,
      "personality": "You are a QA engineer who analyzes code execution and identifies issues.",
      "contextType": "clean"
    }
  },
  
  "actions": {
    "implement": {
      "prompt": "Implement a {{1}} that {{2}}. Include proper error handling and documentation."
    },
    "analyzeExecution": {
      "prompt": "Analyze this execution output:\n\n{{1}}\n\nIdentify any issues, errors, or improvements needed."
    },
    "fixIssues": {
      "prompt": "Fix the issues in this code based on the execution analysis:\n\nOriginal Code:\n{{1}}\n\nExecution Analysis:\n{{2}}"
    }
  },
  
  "workflow": [
    {
      "action": "implement",
      "agent": "developer",
      "inputs": ["Python script", "creates a simple visualization using matplotlib"],
      "output": "visualization.py",
      "format": "python"
    },
    {
      "action": "run_python",
      "inputs": ["visualization.py"],
      "output": "execution.log"
    },
    {
      "action": "analyzeExecution",
      "agent": "tester",
      "inputs": ["execution.log"],
      "output": "analysis.md"
    },
    {
      "action": "fixIssues",
      "agent": "developer",
      "inputs": ["visualization.py", "analysis.md"],
      "output": "visualization_fixed.py",
      "format": "python"
    },
    {
      "action": "run_python",
      "inputs": ["visualization_fixed.py"],
      "output": "final_execution.log"
    }
  ],
  
  "config": {
    "outputDirectory": "./code_test_output",
    "logLevel": "info",
    "queryTimeout": 300
  }
}
```

## CLI Usage

```bash
# Run a scenario
python cli.py scenario.json

# Run multiple times (creates separate output directories)
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

### Agent Configuration
- `model`: The Ollama model to use (e.g., "gemma:2b", "llama2", "mistral")
- `temperature`: Creativity level (0.0 to 1.0)
- `personality`: System prompt defining the agent's role
- `contextType`: How conversation history is managed

### Global Configuration
- `logLevel`: Logging verbosity ("debug", "info", "warning", "error")
- `outputDirectory`: Where to save output files
- `queryTimeout`: Maximum seconds to wait for model responses
- `maxContextTokens`: Maximum tokens before trimming conversation history

## Tips & Best Practices

1. **File Extensions**: Always include extensions (.py, .md, .txt) for better file handling
2. **Action Design**: Keep actions focused on a single task for better reusability
3. **Context Management**: Use `clearContext` between major workflow phases to avoid confusion
4. **Temperature Settings**: Lower temperatures (0.3) for analytical tasks, higher (0.7+) for creative tasks
5. **Error Handling**: The system validates agent and action names before execution
6. **Output Organization**: Use descriptive filenames to track workflow progress
7. **Python Testing**: Use `run_python` to validate generated code automatically
8. **Virtual Environment**: Keep `testenv` isolated with only necessary packages

## Advanced Example: Competitive Development with Testing

```json
{
  "metadata": {
    "name": "Competitive Development with Testing",
    "version": "1.0"
  },
  
  "agents": {
    "dev1": {
      "model": "gemma:2b",
      "temperature": 0.8,
      "personality": "You favor creative, elegant solutions.",
      "contextType": "rolling"
    },
    "dev2": {
      "model": "gemma:2b",
      "temperature": 0.6,
      "personality": "You favor simple, performant solutions.",
      "contextType": "rolling"
    },
    "judge": {
      "model": "gemma:2b",
      "temperature": 0.3,
      "personality": "You evaluate code solutions objectively.",
      "contextType": "clean"
    },
    "tester": {
      "model": "gemma:2b",
      "temperature": 0.4,
      "personality": "You analyze execution results and identify runtime issues.",
      "contextType": "clean"
    }
  },
  
  "actions": {
    "solve": {
      "prompt": "Solve this problem: {{1}}\n\nFocus on your strengths."
    },
    "evaluate": {
      "prompt": "Compare these two solutions:\n\nSolution 1:\n{{1}}\n\nSolution 2:\n{{2}}\n\nWhich is better and why?"
    },
    "analyzeExecution": {
      "prompt": "Analyze this execution output:\n\n{{1}}\n\nReport on performance, errors, and functionality."
    }
  },
  
  "workflow": [
    {
      "action": "loop",
      "iterations": 3,
      "steps": [
        {
          "action": "solve",
          "agent": "dev1",
          "inputs": ["implement a sorting algorithm (round {{iteration}})"],
          "output": "solution1_round{{iteration}}.py",
          "format": "python"
        },
        {
          "action": "solve", 
          "agent": "dev2",
          "inputs": ["implement a sorting algorithm (round {{iteration}})"],
          "output": "solution2_round{{iteration}}.py",
          "format": "python"
        },
        {
          "action": "run_python",
          "inputs": ["solution1_round{{iteration}}.py"],
          "output": "execution1_round{{iteration}}.log"
        },
        {
          "action": "run_python",
          "inputs": ["solution2_round{{iteration}}.py"],
          "output": "execution2_round{{iteration}}.log"
        },
        {
          "action": "analyzeExecution",
          "agent": "tester",
          "inputs": ["execution1_round{{iteration}}.log"],
          "output": "analysis1_round{{iteration}}.md"
        },
        {
          "action": "analyzeExecution",
          "agent": "tester",
          "inputs": ["execution2_round{{iteration}}.log"],
          "output": "analysis2_round{{iteration}}.md"
        },
        {
          "action": "evaluate",
          "agent": "judge",
          "inputs": ["solution1_round{{iteration}}.py", "solution2_round{{iteration}}.py"],
          "output": "evaluation_round{{iteration}}.md"
        }
      ]
    }
  ]
}
```

## Troubleshooting

### File Not Found Errors
- Ensure the file was created in a previous step
- Check that you're using the correct filename with extension
- Verify the output directory path

### Input Count Mismatch
- Count the {{1}}, {{2}}, etc. placeholders in your action prompt
- Ensure your inputs list has exactly that many items

### Context Window Errors
- Use `contextType: "rolling"` for long conversations
- Add `clearContext` actions between major phases
- Reduce `maxContextTokens` in config if needed

### Model Loading Issues
- Ensure the model is installed: `ollama pull model_name`
- Check that Ollama is running: `ollama serve`
- Verify model names match exactly (case-sensitive)

### Python Execution Issues
- Ensure `testenv` virtual environment exists in project directory
- Install required packages in testenv: `source testenv/bin/activate && pip install package_name`
- Check execution logs for specific error messages
- Verify Python script syntax before execution

### Virtual Environment Setup
```bash
# Create testenv
python -m venv testenv

# Activate and install common packages
source testenv/bin/activate
pip install matplotlib numpy pandas requests beautifulsoup4
deactivate
```
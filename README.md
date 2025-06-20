# Ollama Multi-Agent Conversation Manager

A Python script for managing intelligent conversations with Ollama LLMs, supporting self-reflection and multi-agent debate modes.

## Features

- **Self-Reflection Mode**: Automatically follows up on responses with reflection prompts to simulate deeper thinking
- **Debate Mode**: Two agents (creator and judge) iterate on solutions through feedback cycles
- **Configurable**: Easy YAML configuration for agents, prompts, and behavior
- **Async Support**: Efficient handling of multiple conversations

## Installation

1. Ensure Ollama is running on your system:
   ```bash
   ollama serve
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Pull your desired model(s):
   ```bash
   ollama pull gemma2:2b
   ```

## Usage

### Command Line Interface

```bash
# Self-reflection mode (default)
python cli.py "Write a function to sort a list"

# Debate mode
python cli.py "Design a REST API for a todo app" --mode debate

# Save output to file
python cli.py "Explain quantum computing" --output results.json

# Use specific agent
python cli.py "Calculate fibonacci" --agent thinker
```

### Python API

```python
import asyncio
from ollama_agent import ConversationManager

async def example():
    manager = ConversationManager()
    
    # Self-reflection mode
    result = await manager.run_task(
        "Write a Python decorator", 
        mode="self_reflection"
    )
    
    # Debate mode
    result = await manager.run_task(
        "Design a database schema", 
        mode="debate"
    )

asyncio.run(example())
```

## Configuration

Edit `config.yaml` to customize:

- **agents**: Define multiple agents with different models and prompts
- **reflection_prompts**: Customize follow-up prompts for self-reflection
- **debate**: Configure creator/judge agents and number of rounds

### Example: Adding a New Agent

```yaml
agents:
  - name: "coder"
    model: "codellama:7b"
    temperature: 0.3
    system_prompt: "You are an expert programmer focused on clean, efficient code."
```

## Notes

- The script uses the official `ollama` Python package
- MCP (Model Context Protocol) is not directly supported by Ollama, but this script achieves similar multi-turn reasoning through structured prompts
- Results are saved with timestamps and full conversation history
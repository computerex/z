# Agentic Coding Harness

An agentic coding harness optimized for Z.AI GLM-4.7 models.

## Features

- **Tool Calling**: Reliable function calling with proper error handling
- **File Operations**: Read, write, and edit files with precision
- **Code Editing**: String-replace based editing with context matching
- **File Search**: Glob-based file discovery
- **Shell Commands**: Execute shell commands with output/error capture
- **Indexing**: Automatic code indexing for search
- **Lexical Search**: Fast text/regex search across codebase
- **Semantic Search**: Natural language code search using embeddings
- **Context Compaction**: Automatic context management to stay within limits

## Installation

```bash
cd harness
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file:

```
LLM_API_URL=https://api.z.ai/api/paas/v4/
LLM_API_KEY=your-api-key
LLM_MODEL=glm-4.7
```

## Usage

### Interactive Mode

```bash
python -m harness.cli
```

### Single Command

```bash
python -m harness.cli -m "Read the README.md and summarize it"
```

### Options

- `-e, --env`: Path to .env file (default: .env)
- `-w, --workspace`: Workspace directory (default: current)
- `-m, --message`: Single message to run
- `--max-iterations`: Maximum iterations (default: 20)
- `--max-tokens`: Maximum context tokens (default: 32000)

## Running Tests

```bash
# Run all unit tests
pytest

# Run with coverage
pytest --cov=harness

# Run integration tests (requires API key)
pytest -m integration

# Run specific test file
pytest tests/test_file_tools.py
```

## Architecture

```
src/harness/
├── __init__.py      # Package exports
├── agent.py         # Main Agent class
├── cli.py           # Command-line interface
├── config.py        # Configuration management
├── context.py       # Context window management
├── llm_client.py    # LLM API client
└── tools/
    ├── __init__.py
    ├── registry.py      # Tool registry
    ├── file_tools.py    # File operations
    ├── shell_tools.py   # Shell commands
    └── search_tools.py  # Search functionality
```

## Tools Available

| Tool | Description |
|------|-------------|
| `read_file` | Read file content, optionally with line ranges |
| `write_file` | Write content to a file |
| `edit_file` | Replace a specific string in a file |
| `list_directory` | List directory contents |
| `file_search` | Search for files by glob pattern |
| `run_command` | Execute shell commands |
| `lexical_search` | Text/regex search in files |
| `semantic_search` | Natural language code search |

## License

MIT

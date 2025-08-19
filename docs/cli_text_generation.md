# CortexSDR Text Generation CLI

A command-line interface for text generation using CortexSDR models, similar to Ollama.

## Features

- **Interactive Chat Mode**: Chat with your SDR model in real-time
- **Single Prompt Mode**: Generate text from a single prompt
- **Model Loading**: Load compressed SDR models (.sdr files)
- **Configurable Parameters**: Set maximum output length and other parameters
- **Signal Handling**: Graceful shutdown with Ctrl+C

## Usage

### Basic Usage

```bash
# Interactive mode (default)
./cortexsdr_text_cli model.sdr

# Single prompt mode
./cortexsdr_text_cli -p "Hello, how are you?" model.sdr

# Set maximum output length
./cortexsdr_text_cli -m 200 -p "Tell me a story" model.sdr
```

### Command Line Options

- `-h, --help`: Show help message
- `-v, --version`: Show version information
- `-i, --interactive`: Start interactive chat mode
- `-p, --prompt TEXT`: Generate text from prompt and exit
- `-m, --max-length N`: Maximum output length (default: 100)

### Examples

```bash
# Start interactive chat
./cortexsdr_text_cli -i model.sdr

# Generate a short response
./cortexsdr_text_cli -p "What is AI?" model.sdr

# Generate a longer story
./cortexsdr_text_cli -m 500 -p "Write a short story about a robot" model.sdr

# Show help
./cortexsdr_text_cli --help
```

## Interactive Mode

When running in interactive mode, you can:

- Type your messages and press Enter
- Type `quit` or `exit` to close the application
- Use Ctrl+C for graceful shutdown

Example session:
```
=== CortexSDR Text Generation CLI ===
Type your messages and press Enter. Type 'quit' to exit.
=====================================

> Hello, how are you?
Generating response...
Generated in 45ms
Response: Hello! I'm doing well, thank you for asking. How are you today?

> Tell me about AI
Generating response...
Generated in 67ms
Response: Artificial Intelligence (AI) is a branch of computer science...

> quit
Shutting down...
```

## Model Requirements

The CLI expects compressed SDR models with the `.sdr` extension. These models should be:

- Compressed using the CortexSDR compression tools
- Compatible with the SDR inference engine
- Optimized for text generation tasks

## Building

The CLI is built as part of the main CortexSDR build process:

```bash
# Build the entire project
./build_desktop.sh

# The CLI will be available as:
./build_desktop/cortexsdr_text_cli
```

## Error Handling

The CLI includes comprehensive error handling:

- Model loading errors
- Inference failures
- Invalid command line arguments
- Signal handling for graceful shutdown

## Performance

The CLI is optimized for:

- Fast model loading
- Efficient text generation
- Low memory usage
- Real-time interaction

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model file exists and has the correct path
2. **Permission denied**: Make sure the model file is readable
3. **Inference errors**: Check that the model is compatible with the current SDK version

### Debug Mode

For debugging, you can compile with debug flags:

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make cortexsdr_text_cli
```

## Integration

The CLI can be easily integrated into scripts and other applications:

```bash
# Use in a shell script
response=$(./cortexsdr_text_cli -p "Generate a summary" model.sdr)
echo "Generated: $response"

# Pipe input/output
echo "What is machine learning?" | ./cortexsdr_text_cli model.sdr
```

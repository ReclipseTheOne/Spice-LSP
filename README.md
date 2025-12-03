# Spice Language Server

Language Server Protocol (LSP) implementation for the Spice programming language.

## Features

- Real-time syntax checking and diagnostics
- Code completion for keywords and snippets
- Hover information for language constructs
- Integration with Spice Lexer, Parser, and Type system

## Installation

```bash
pip install -e .
```

## Usage

The LSP server is automatically started by the VSCode extension. To run it manually:

```bash
spice-lsp
```

## Architecture

The server integrates with:
- `spice.lexer.Lexer` - Tokenization
- `spice.parser.Parser` - AST generation
- `spice.compilation.pipeline.CompilationPipeline` - Full compilation

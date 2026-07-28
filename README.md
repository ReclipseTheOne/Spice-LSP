# Spice Language Server

Language Server Protocol (LSP) implementation for the Spice programming language.

## Features

- Diagnostics from the compiler's own checks - types, interfaces, overloads,
  generic bounds and final reassignment - **resolved across module boundaries**
- Go to definition, including names that arrived on an import
- Completion for keywords, snippets, compile-time annotations, imports, and
  every symbol the file can actually name
- Hover for keywords, annotations, scientific-notation literals, and declared
  symbols (with the module they came from)
- Document symbols, for the outline and breadcrumbs

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

The server does not resolve modules itself. It builds a `SpiceProject` - the
compiler's own model of a build - and asks it the same questions the compiler
asks:

- `spice_lsp.workspace.SpiceWorkspace` - one `SpiceProject` per root, cached
  between requests and dropped when a file is saved or changes on disk
- `spice.compilation.SpiceProject` - the module graph, lookup paths, and
  symbols visible from any given file
- `spice.compilation.checks` - the checks, run over the edited buffer once its
  imports have been parsed

That last part is the point. Running the checks against a single file made a
class implementing an interface declared in another module report as
implementing an unknown one, in code that compiles.

### Configuration

Sent by the client as `initializationOptions`:

| Option | Meaning |
| --- | --- |
| `lookupPaths` | Extra directories to resolve module names against, on top of the workspace folders and the interpreter's own paths. |

### Notes for anything embedding this

- The compiler's stage logging writes to **stdout**, which is the LSP's
  JSON-RPC channel. `spice_lsp.workspace` silences it on import; anything that
  drives the compiler alongside a stdio server has to do the same.
- Requires a `spice-lang` new enough to provide `spice.compilation.SpiceProject`.

## Tests

```bash
pytest tests/ -q
```

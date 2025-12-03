"""
Spice Language Server Protocol Implementation.

This LSP server integrates with the Spice compiler's Lexer, Parser, and Type system
to provide rich IDE features for .spc files.
"""

import logging
from typing import List, Optional
from pathlib import Path
import sys

# Add parent directory to path to import spice package
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "spice-lang"))

from pygls.server import LanguageServer
from pygls.lsp.types import (
    Diagnostic,
    DiagnosticSeverity,
    Position,
    Range,
    CompletionItem,
    CompletionItemKind,
    CompletionList,
    CompletionParams,
    Hover,
    HoverParams,
    MarkupContent,
    MarkupKind,
    DidOpenTextDocumentParams,
    DidChangeTextDocumentParams,
    DidSaveTextDocumentParams,
    TextDocumentPositionParams,
    Location,
)

from spice.lexer import Lexer, TokenType
from spice.parser import Parser
from spice.compilation.pipeline import CompilationPipeline
from spice.errors import SpiceError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Spice Language Server
server = LanguageServer("spice-lsp", "v0.1")


class SpiceDocument:
    """Represents a parsed Spice document with AST and diagnostics."""

    def __init__(self, uri: str, source: str):
        self.uri = uri
        self.source = source
        self.diagnostics: List[Diagnostic] = []
        self.ast = None
        self.tokens = []
        self.parse()

    def parse(self):
        """Parse the document and collect diagnostics."""
        try:
            # Tokenize
            lexer = Lexer()
            self.tokens = lexer.tokenize(self.source)

            # Check for lexer errors
            if lexer.errors:
                for error in lexer.errors:
                    self.diagnostics.append(
                        Diagnostic(
                            range=Range(
                                start=Position(line=error.line - 1, character=error.column),
                                end=Position(line=error.line - 1, character=error.column + 1)
                            ),
                            message=str(error),
                            severity=DiagnosticSeverity.Error,
                            source="spice-lexer"
                        )
                    )

            # Parse
            parser = Parser()
            parser.tokens = self.tokens
            self.ast = parser.parse()

        except SpiceError as e:
            # Handle Spice-specific errors
            line = getattr(e, 'line', 0)
            column = getattr(e, 'column', 0)

            self.diagnostics.append(
                Diagnostic(
                    range=Range(
                        start=Position(line=max(0, line - 1), character=max(0, column)),
                        end=Position(line=max(0, line - 1), character=max(0, column) + 10)
                    ),
                    message=str(e),
                    severity=DiagnosticSeverity.Error,
                    source="spice-parser"
                )
            )
        except SyntaxError as e:
            # Handle syntax errors
            self.diagnostics.append(
                Diagnostic(
                    range=Range(
                        start=Position(line=0, character=0),
                        end=Position(line=0, character=10)
                    ),
                    message=str(e),
                    severity=DiagnosticSeverity.Error,
                    source="spice-syntax"
                )
            )
        except Exception as e:
            logger.exception(f"Unexpected error parsing document: {e}")
            self.diagnostics.append(
                Diagnostic(
                    range=Range(
                        start=Position(line=0, character=0),
                        end=Position(line=0, character=10)
                    ),
                    message=f"Internal error: {str(e)}",
                    severity=DiagnosticSeverity.Error,
                    source="spice-internal"
                )
            )


# Document cache
documents: dict[str, SpiceDocument] = {}


@server.feature("textDocument/didOpen")
def did_open(ls: LanguageServer, params: DidOpenTextDocumentParams):
    """Handle document open event."""
    uri = params.text_document.uri
    source = params.text_document.text

    logger.info(f"Document opened: {uri}")

    # Parse and cache document
    doc = SpiceDocument(uri, source)
    documents[uri] = doc

    # Send diagnostics
    ls.publish_diagnostics(uri, doc.diagnostics)


@server.feature("textDocument/didChange")
def did_change(ls: LanguageServer, params: DidChangeTextDocumentParams):
    """Handle document change event."""
    uri = params.text_document.uri

    # Get latest content
    if params.content_changes:
        source = params.content_changes[0].text

        logger.info(f"Document changed: {uri}")

        # Re-parse document
        doc = SpiceDocument(uri, source)
        documents[uri] = doc

        # Send updated diagnostics
        ls.publish_diagnostics(uri, doc.diagnostics)


@server.feature("textDocument/didSave")
def did_save(ls: LanguageServer, params: DidSaveTextDocumentParams):
    """Handle document save event."""
    uri = params.text_document.uri
    logger.info(f"Document saved: {uri}")

    # Could trigger full compilation here if needed
    if uri in documents:
        ls.publish_diagnostics(uri, documents[uri].diagnostics)


@server.feature("textDocument/completion")
def completions(params: CompletionParams) -> Optional[CompletionList]:
    """Provide completion items."""
    uri = params.text_document.uri

    if uri not in documents:
        return None

    doc = documents[uri]
    items = []

    # Spice keywords
    keywords = [
        'interface', 'abstract', 'final', 'static', 'extends', 'implements',
        'def', 'class', 'if', 'elif', 'else', 'for', 'while', 'return',
        'import', 'from', 'as', 'pass', 'break', 'continue', 'switch', 'case', 'default'
    ]

    for keyword in keywords:
        items.append(
            CompletionItem(
                label=keyword,
                kind=CompletionItemKind.Keyword,
                detail="Spice keyword"
            )
        )

    # Add snippets
    items.extend([
        CompletionItem(
            label="interface",
            kind=CompletionItemKind.Snippet,
            detail="Interface declaration",
            insert_text="interface ${1:Name} {\n\tdef ${2:method}(${3:params}) -> ${4:ReturnType};\n}"
        ),
        CompletionItem(
            label="abstract class",
            kind=CompletionItemKind.Snippet,
            detail="Abstract class declaration",
            insert_text="abstract class ${1:Name} {\n\tabstract def ${2:method}() -> ${3:ReturnType};\n}"
        ),
    ])

    return CompletionList(is_incomplete=False, items=items)


@server.feature("textDocument/hover")
def hover(params: HoverParams) -> Optional[Hover]:
    """Provide hover information."""
    uri = params.text_document.uri
    position = params.position

    if uri not in documents:
        return None

    doc = documents[uri]

    # Get word at position
    lines = doc.source.split('\n')
    if position.line >= len(lines):
        return None

    line = lines[position.line]

    # Simple word extraction
    start = position.character
    end = position.character

    while start > 0 and (line[start - 1].isalnum() or line[start - 1] == '_'):
        start -= 1

    while end < len(line) and (line[end].isalnum() or line[end] == '_'):
        end += 1

    word = line[start:end]

    # Provide hover info for keywords
    hover_docs = {
        'interface': '**interface** keyword\n\nDeclares an interface (Protocol in Python) that defines method signatures',
        'abstract': '**abstract** modifier\n\nMarks a class or method as abstract (must be overridden)',
        'final': '**final** modifier\n\nPrevents a class from being inherited or a method from being overridden',
        'static': '**static** modifier\n\nDeclares a static method that belongs to the class rather than instances',
        'extends': '**extends** keyword\n\nSpecifies class inheritance',
        'implements': '**implements** keyword\n\nSpecifies that a class implements one or more interfaces',
    }

    if word in hover_docs:
        return Hover(
            contents=MarkupContent(
                kind=MarkupKind.Markdown,
                value=hover_docs[word]
            )
        )

    return None


def start_server():
    """Start the Spice LSP server."""
    logger.info("Starting Spice Language Server")
    server.start_io()


if __name__ == "__main__":
    start_server()

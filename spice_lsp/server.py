"""
Spice Language Server Protocol Implementation.

This LSP server integrates with the Spice compiler's Lexer, Parser, and Type system
to provide rich IDE features for .spc files.
"""

import logging
import sys
from typing import List, Optional
from pathlib import Path

from pygls.lsp.server import LanguageServer
from lsprotocol.types import (
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
    PublishDiagnosticsParams,
    TextDocumentSyncKind,
)

from spice.lexer import Lexer, TokenType
from spice.parser import Parser
from spice.compilation.spicefile import SpiceFile
from spice.compilation.checks import SymbolTableBuilder, TypeChecker, MethodOverloadResolver, InterfaceChecker, FinalChecker, CheckError
from spice.errors import SpiceError

# Set up file logging for debugging
LOG_FILE = Path.home() / ".spice-lsp.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stderr),
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Spice LSP starting, logging to {LOG_FILE}")

# Spice Language Server
server = LanguageServer("spice-lsp", "v0.1", text_document_sync_kind=TextDocumentSyncKind.Full)


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
            logger.debug(f"Parsing document: {self.uri}")

            # Tokenize
            lexer = Lexer()
            self.tokens = lexer.tokenize(self.source)
            logger.debug(f"Tokenized {len(self.tokens)} tokens")

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
            self.ast = parser.parse(self.tokens)
            logger.debug(f"Parsed AST with {len(self.ast.body) if self.ast else 0} top-level statements")

            # Run semantic checks if parsing succeeded
            if self.ast and not self.diagnostics:
                self._run_semantic_checks()

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

    def _make_diagnostic(self, error, source: str) -> Diagnostic:
        """Create a diagnostic from an error, extracting line/column if available."""
        # Handle CheckError with line/column
        if isinstance(error, CheckError):
            logger.debug(f"CheckError received: line={error.line}, column={error.column}, message={error.message}")
            line = max(0, error.line - 1)  # LSP uses 0-indexed lines
            column = max(0, error.column)
            message = f"{error.message} ({error.line}:{error.column})"
        # Handle errors with line/column attributes
        elif hasattr(error, 'line') and hasattr(error, 'column'):
            err_line = getattr(error, 'line', 1)
            err_col = getattr(error, 'column', 0)
            line = max(0, err_line - 1)
            column = max(0, err_col)
            message = f"{error} ({err_line}:{err_col})"
        # Fallback for string errors
        else:
            line = 0
            column = 0
            message = str(error)

        return Diagnostic(
            range=Range(
                start=Position(line=line, character=column),
                end=Position(line=line, character=column + 10)
            ),
            message=message,
            severity=DiagnosticSeverity.Error,
            source=source
        )

    def _run_semantic_checks(self):
        """Run semantic checks on the parsed AST."""
        try:
            spice_file = SpiceFile.empty(self.source)
            spice_file.ast = self.ast

            logger.debug("Building symbol table...")
            symbol_builder = SymbolTableBuilder()
            symbol_builder.check(spice_file)
            logger.debug(f"Symbol table built: {spice_file.symbol_table is not None}")

            logger.debug("Checking method overloads...")
            overload_resolver = MethodOverloadResolver()
            if not overload_resolver.check(spice_file):
                for error in overload_resolver.errors:
                    self.diagnostics.append(self._make_diagnostic(error, "spice-overload"))

            logger.debug("Running type checker...")
            type_checker = TypeChecker()
            if not type_checker.check(spice_file):
                for error in type_checker.errors:
                    self.diagnostics.append(self._make_diagnostic(error, "spice-type"))

            logger.debug("Checking interface implementations...")
            interface_checker = InterfaceChecker()
            if not interface_checker.check(spice_file):
                for error in interface_checker.errors:
                    self.diagnostics.append(self._make_diagnostic(error, "spice-interface"))

            logger.debug("Checking final constraints...")
            final_checker = FinalChecker()
            if not final_checker.check(spice_file):
                for error in final_checker.errors:
                    self.diagnostics.append(self._make_diagnostic(error, "spice-final"))

            logger.debug(f"Semantic checks complete, {len(self.diagnostics)} diagnostics")

        except Exception as e:
            logger.exception(f"Error during semantic checks: {e}")


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
    ls.text_document_publish_diagnostics(PublishDiagnosticsParams(uri=uri, diagnostics=doc.diagnostics))


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
        ls.text_document_publish_diagnostics(PublishDiagnosticsParams(uri=uri, diagnostics=doc.diagnostics))


@server.feature("textDocument/didSave")
def did_save(ls: LanguageServer, params: DidSaveTextDocumentParams):
    """Handle document save event."""
    uri = params.text_document.uri
    logger.info(f"Document saved: {uri}")

    # Could trigger full compilation here if needed
    if uri in documents:
        ls.text_document_publish_diagnostics(PublishDiagnosticsParams(uri=uri, diagnostics=documents[uri].diagnostics))


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

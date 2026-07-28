"""
Spice Language Server Protocol Implementation.

This LSP server integrates with the Spice compiler's Lexer, Parser, and Type system
to provide rich IDE features for .spc files.
"""

import logging
import sys
import re
import sysconfig
import importlib
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Dict, Tuple
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
    DefinitionParams,
    DocumentSymbol,
    DocumentSymbolParams,
    SymbolKind,
    DidChangeWatchedFilesParams,
    InitializeParams,
)

from spice.lexer import Lexer
from spice.parser import Parser
from spice.parser.ast_nodes import (
    ClassDeclaration,
    FunctionDeclaration,
    InterfaceDeclaration,
    DataClassDeclaration,
    EnumDeclaration,
)
from spice.compilation.spicefile import SpiceFile
from spice.compilation.checks import CheckError
from spice.errors import SpiceError

import spice.annotations.builtins
from spice.annotations import all_processors

from spice_lsp.workspace import SpiceWorkspace, path_to_uri, uri_to_path

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

# Every project the editor has a file open in. Module resolution and the
# compile-time checks both go through this, so the server and the compiler
# answer the same question the same way.
workspace = SpiceWorkspace()


class SpiceDocument:
    """One open Spice document, checked against the project it belongs to."""

    def __init__(self, uri: str, source: str):
        self.uri = uri
        self.source = source
        self.path = uri_to_path(uri)
        self.diagnostics: List[Diagnostic] = []
        self.ast = None
        self.tokens = []
        self.spice_file: Optional[SpiceFile] = None
        self.parse()

    @property
    def project(self):
        """The build this document sits in, or None if it was never analyzed."""
        return getattr(self.spice_file, "project", None)

    def parse(self):
        """Parse the document and collect diagnostics."""
        try:
            logger.debug(f"Parsing document: {self.uri}")

            if self.path is None:
                # Not a file on disk - an untitled buffer, say. Nothing can be
                # resolved against it, so check what there is on its own.
                self._parse_standalone()
                return

            analysis = workspace.analyze(self.path, self.source)

            self.spice_file = analysis.file
            self.ast = analysis.file.ast
            self.tokens = analysis.file.tokens

            for source, error in analysis.errors:
                self.diagnostics.append(self._make_diagnostic(error, source))
            for source, warning in analysis.warnings:
                self.diagnostics.append(
                    self._make_diagnostic(warning, source, DiagnosticSeverity.Warning)
                )

            logger.debug(f"Analysis complete, {len(self.diagnostics)} diagnostics")

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

    def _parse_standalone(self):
        """Check a buffer with no path of its own.

        Nothing here can resolve an import, so this is the old single-file
        behaviour: what the file declares is all it knows about.
        """
        self.tokens = Lexer().tokenize(self.source)
        self.ast = Parser().parse(self.tokens)

        self.spice_file = SpiceFile.empty(self.source)
        self.spice_file.ast = self.ast
        self.spice_file.tokens = self.tokens

        from spice.compilation.checks import SymbolTableBuilder
        SymbolTableBuilder().check(self.spice_file)

    def _make_diagnostic(self, error, source: str,
                         severity: DiagnosticSeverity = DiagnosticSeverity.Error) -> Diagnostic:
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
            severity=severity,
            source=source
        )


# Document cache
documents: dict[str, SpiceDocument] = {}

# Module discovery caches. Both are dropped whenever the workspace is, since
# what a module exports is exactly what changes when a file is saved.
_module_cache: Dict[str, Dict] = {}  # uri -> {stdlib, packages, spice, all}
_export_cache: Dict[str, List[Tuple[str, CompletionItemKind]]] = {}  # module_name -> [(name, kind)]


def invalidate_caches() -> None:
    """Forget everything read off disk."""
    workspace.invalidate()
    _module_cache.clear()
    _export_cache.clear()


def _uri_to_path(uri: str) -> Optional[Path]:
    """Convert a file:// URI to a Path."""
    return uri_to_path(uri)


def get_lookup_paths(document_uri: str) -> List[Path]:
    """Where module names are looked up for this document.

    The project's own list, in the project's own order. This used to be a
    second implementation living beside the compiler's, and the two disagreed
    about which directory a bare module name resolves against.
    """
    doc_path = _uri_to_path(document_uri)
    if doc_path is None:
        return []

    project = workspace.project_for(doc_path)
    file = project.modules.get(doc_path.resolve())
    if file is not None:
        return list(project.lookup_paths_for(file))
    return [doc_path.parent] + list(project.base_lookup_paths)


def scan_for_modules(document_uri: str) -> Dict:
    """Scan paths for available modules with caching."""
    if document_uri in _module_cache:
        return _module_cache[document_uri]

    logger.debug(f"Scanning modules for: {document_uri}")

    modules = {
        "stdlib": [],      # Python standard library
        "packages": [],    # Installed packages
        "spice": [],       # Local .spc files
        "all": [],         # Combined
    }

    for name in sys.builtin_module_names:
        if not name.startswith('_'):
            modules["stdlib"].append((name, "builtin"))

    paths = get_lookup_paths(document_uri)

    stdlib_path_str = sysconfig.get_path('stdlib') or ""

    for path in paths:
        if not path.exists():
            continue

        try:
            is_stdlib = stdlib_path_str and str(path.resolve()).startswith(stdlib_path_str)
            is_local = path == _uri_to_path(document_uri).parent if _uri_to_path(document_uri) else False

            for item in path.iterdir():
                if item.name.startswith(('_', '.')):
                    continue

                if item.suffix == '.spc':
                    modules["spice"].append((item.stem, "spice"))

                elif item.suffix == '.py':
                    category = "stdlib" if is_stdlib else "packages"
                    modules[category].append((item.stem, "python"))

                elif item.is_dir():
                    if (item / '__init__.py').exists():
                        category = "stdlib" if is_stdlib else "packages"
                        modules[category].append((item.name, "package"))
                    elif (item / '__init__.spc').exists():
                        modules["spice"].append((item.name, "spice_package"))

        except PermissionError:
            continue
        except Exception as e:
            logger.debug(f"Error scanning {path}: {e}")

    for key in ["stdlib", "packages", "spice"]:
        modules[key] = list(set(modules[key]))

    modules["all"] = (
        [(name, typ) for name, typ in modules["spice"]] +
        [(name, typ) for name, typ in modules["packages"]] +
        [(name, typ) for name, typ in modules["stdlib"]]
    )

    logger.debug(f"Found {len(modules['spice'])} spice, {len(modules['packages'])} packages, {len(modules['stdlib'])} stdlib modules")

    _module_cache[document_uri] = modules
    return modules


def get_module_exports(module_name: str, document_uri: str) -> List[Tuple[str, CompletionItemKind]]:
    """Get exports from a module (lazy import for Python, parse for Spice)."""
    cache_key = f"{document_uri}:{module_name}"
    if cache_key in _export_cache:
        return _export_cache[cache_key]

    logger.debug(f"Getting exports for module: {module_name}")
    exports: List[Tuple[str, CompletionItemKind]] = []

    doc_path = _uri_to_path(document_uri)
    if doc_path:
        # Check for .spc file in dir
        spc_path = doc_path.parent / f"{module_name}.spc"
        if spc_path.exists():
            exports = _get_spice_file_exports(spc_path)
            _export_cache[cache_key] = exports
            return exports

        # Check for .spc package (need to implement __init__ logic for modules first :p)
        spc_pkg_path = doc_path.parent / module_name / "__init__.spc"
        if spc_pkg_path.exists():
            exports = _get_spice_file_exports(spc_pkg_path)
            _export_cache[cache_key] = exports
            return exports

    exports = _get_python_module_exports(module_name)
    _export_cache[cache_key] = exports
    return exports


def _get_python_module_exports(module_name: str) -> List[Tuple[str, CompletionItemKind]]:
    """Dynamically import a Python module and get its exports."""
    exports: List[Tuple[str, CompletionItemKind]] = []

    try:
        module = importlib.import_module(module_name)

        for name in dir(module):
            if name.startswith('_'):
                continue

            try:
                obj = getattr(module, name, None)
                if obj is None:
                    kind = CompletionItemKind.Variable
                elif isinstance(obj, type):
                    kind = CompletionItemKind.Class
                elif callable(obj):
                    kind = CompletionItemKind.Function
                else:
                    kind = CompletionItemKind.Variable

                exports.append((name, kind))
            except Exception:
                exports.append((name, CompletionItemKind.Variable))

        logger.debug(f"Found {len(exports)} exports from Python module {module_name}")

    except Exception as e:
        logger.debug(f"Failed to import {module_name}: {e}")

    return exports


def _get_spice_file_exports(file_path: Path) -> List[Tuple[str, CompletionItemKind]]:
    """Parse a .spc file and extract its exports (classes, functions, interfaces)."""
    exports: List[Tuple[str, CompletionItemKind]] = []

    try:
        source = file_path.read_text(encoding='utf-8')
        lexer = Lexer()
        tokens = lexer.tokenize(source)
        parser = Parser()
        ast = parser.parse(tokens)

        for node in ast.body:
            if isinstance(node, ClassDeclaration):
                exports.append((node.name, CompletionItemKind.Class))
            elif isinstance(node, DataClassDeclaration):
                exports.append((node.name, CompletionItemKind.Class))
            elif isinstance(node, EnumDeclaration):
                exports.append((node.name, CompletionItemKind.Enum))
            elif isinstance(node, FunctionDeclaration):
                exports.append((node.name, CompletionItemKind.Function))
            elif isinstance(node, InterfaceDeclaration):
                exports.append((node.name, CompletionItemKind.Interface))

        logger.debug(f"Found {len(exports)} exports from Spice file {file_path}")

    except Exception as e:
        logger.debug(f"Failed to parse {file_path}: {e}")

    return exports


def detect_import_context(source: str, position: Position) -> Dict:
    """Detect if cursor is in an import context and what type.

    Returns:
        {
            "in_import": bool,
            "type": "import" | "from_module" | "from_names" | None,
            "partial": str,  # What user has typed so far
            "module": str | None,  # For 'from X import', the module name
        }
    """
    result = {
        "in_import": False,
        "type": None,
        "partial": "",
        "module": None,
    }

    lines = source.split('\n')
    if position.line >= len(lines):
        return result

    line = lines[position.line]
    line_before_cursor = line[:position.character]

    # "from module import name" - cursor after import
    from_import_match = re.match(r'^from\s+([\w.]+)\s+import\s+(\w*)$', line_before_cursor)
    if from_import_match:
        result["in_import"] = True
        result["type"] = "from_names"
        result["module"] = from_import_match.group(1)
        result["partial"] = from_import_match.group(2)
        return result

    # "from module import name, name2" - cursor after comma
    from_import_multi_match = re.match(r'^from\s+([\w.]+)\s+import\s+[\w\s,]+,\s*(\w*)$', line_before_cursor)
    if from_import_multi_match:
        result["in_import"] = True
        result["type"] = "from_names"
        result["module"] = from_import_multi_match.group(1)
        result["partial"] = from_import_multi_match.group(2)
        return result

    # "from partial" - cursor after from
    from_module_match = re.match(r'^from\s+(\w*)$', line_before_cursor)
    if from_module_match:
        result["in_import"] = True
        result["type"] = "from_module"
        result["partial"] = from_module_match.group(1)
        return result

    # "import partial" - cursor after import
    import_match = re.match(r'^import\s+([\w.]*)$', line_before_cursor)
    if import_match:
        result["in_import"] = True
        result["type"] = "import"
        result["partial"] = import_match.group(1)
        return result

    # "import mod1, partial" - cursor after comma in import
    import_multi_match = re.match(r'^import\s+[\w.\s,]+,\s*([\w.]*)$', line_before_cursor)
    if import_multi_match:
        result["in_import"] = True
        result["type"] = "import"
        result["partial"] = import_multi_match.group(1)
        return result

    return result


def get_import_completions(uri: str, context: Dict) -> CompletionList:
    """Generate completion items for import statements."""
    items: List[CompletionItem] = []

    if context["type"] in ("import", "from_module"):
        # Suggest module names
        modules = scan_for_modules(uri)
        partial = context["partial"].lower()

        for mod_name, mod_type in modules["all"]:
            if partial and not mod_name.lower().startswith(partial):
                continue

            # Sort by relevancy
            if mod_type in ("spice", "spice_package"):
                sort_prefix = "0"
                detail = "Spice module"
            elif mod_type == "builtin":
                sort_prefix = "2"
                detail = "Python builtin"
            elif mod_type == "package":
                sort_prefix = "1"
                detail = "Python package"
            else:
                sort_prefix = "1"
                detail = "Python module"

            items.append(CompletionItem(
                label=mod_name,
                kind=CompletionItemKind.Module,
                detail=detail,
                sort_text=f"{sort_prefix}{mod_name}",
            ))

    elif context["type"] == "from_names":
        # Suggest exports from the specified module
        module_name = context["module"]
        partial = context["partial"].lower()

        exports = get_module_exports(module_name, uri)

        for name, kind in exports:
            if partial and not name.lower().startswith(partial):
                continue

            items.append(CompletionItem(
                label=name,
                kind=kind,
                detail=f"from {module_name}",
                sort_text=name,
            ))

    return CompletionList(is_incomplete=False, items=items)


def detect_import_definition_context(source: str, position: Position) -> Optional[Dict]:
    """Detect if cursor is on an import and what part.

    Returns:
        {
            "type": "module" | "name",
            "module": str,  # The module being imported
            "name": str | None,  # The specific name being imported (for 'from X import Y')
        }
        or None if not on an import
    """
    lines = source.split('\n')
    if position.line >= len(lines):
        return None

    line = lines[position.line]
    col = position.character

    # Extract word at cursor
    start = col
    end = col
    while start > 0 and (line[start - 1].isalnum() or line[start - 1] == '_' or line[start - 1] == '.'):
        start -= 1
    while end < len(line) and (line[end].isalnum() or line[end] == '_' or line[end] == '.'):
        end += 1

    word = line[start:end]
    if not word:
        return None

    # Check for "from module import name" pattern
    from_import_match = re.match(r'^from\s+([\w.]+)\s+import\s+(.+)$', line)
    if from_import_match:
        module = from_import_match.group(1)
        names_part = from_import_match.group(2)

        # Find where the module name is in the line
        module_start = line.find(module, 5)  # After "from "
        module_end = module_start + len(module)

        # Check if cursor is on the module name
        if module_start <= col <= module_end:
            return {"type": "module", "module": module, "name": None}

        # Check if cursor is on one of the imported names
        # Parse the names (handling "name as alias, name2 as alias2" etc.)
        import_start = line.find("import ") + 7
        if col >= import_start:

            # Find which name the cursor is on
            names_str = names_part
            current_pos = import_start
            for part in names_str.split(','):
                part = part.strip()

                # Handle "name as alias"
                name_match = re.match(r'^(\w+)(?:\s+as\s+\w+)?', part)
                if name_match:
                    name = name_match.group(1)
                    name_pos = line.find(name, current_pos)
                    if name_pos != -1 and name_pos <= col < name_pos + len(name):
                        return {"type": "name", "module": module, "name": name}
                current_pos = line.find(part, current_pos) + len(part)

        return None

    # Check for "import module" pattern
    import_match = re.match(r'^import\s+([\w.]+)(?:\s+as\s+\w+)?', line)
    if import_match:
        module = import_match.group(1)
        module_start = line.find(module, 7)  # After "import "
        module_end = module_start + len(module)

        if module_start <= col <= module_end:
            return {"type": "module", "module": module, "name": None}

    return None


def resolve_module_path(module_name: str, document_uri: str) -> Optional[Path]:
    """Resolve a module name to its file path, the way a build would.

    Handed to the project rather than repeated here, so a jump-to-definition on
    an import line lands on the file the compiler would actually have read.
    """
    doc_path = _uri_to_path(document_uri)
    if doc_path is None:
        return None

    project = workspace.project_for(doc_path)
    file = project.modules.get(doc_path.resolve())
    return project.resolve_module(module_name, file)


def find_symbol_in_file(file_path: Path, symbol_name: str) -> Optional[Tuple[int, int]]:
    """Find a symbol's line and column in a file. Returns (line, column) 0-indexed."""
    try:
        if file_path.suffix == '.spc':
            source = file_path.read_text(encoding='utf-8')
            lexer = Lexer()
            tokens = lexer.tokenize(source)
            parser = Parser()
            ast = parser.parse(tokens)

            for node in ast.body:
                if isinstance(node, (ClassDeclaration, FunctionDeclaration, InterfaceDeclaration,
                                     DataClassDeclaration, EnumDeclaration)):
                    if node.name == symbol_name:
                        # AST nodes use 1-indexed lines
                        return (node.line - 1, node.column)

        elif file_path.suffix == '.py':
            # For Python files, do a simple text search for the definition
            source = file_path.read_text(encoding='utf-8')
            lines = source.split('\n')

            for i, line in enumerate(lines):
                # Look for class or function definitions
                class_match = re.match(rf'^class\s+{re.escape(symbol_name)}\b', line)
                if class_match:
                    return (i, line.find(symbol_name))

                func_match = re.match(rf'^def\s+{re.escape(symbol_name)}\b', line)
                if func_match:
                    return (i, line.find(symbol_name))

                # Also check for simple assignments at module level (constants)
                assign_match = re.match(rf'^{re.escape(symbol_name)}\s*=', line)
                if assign_match:
                    return (i, 0)

    except Exception as e:
        logger.debug(f"Error finding symbol {symbol_name} in {file_path}: {e}")

    return None


def _declaration_location(module: SpiceFile, node: object, name: str) -> Optional[Location]:
    """A Location for `name` declared at `node`, in whichever module owns it.

    The module is the point. Definition used to answer with the current
    document's URI whatever it found, so following an imported class jumped to
    that line number in the file being edited - a different piece of code.
    """
    line = getattr(node, "line", None)
    if line is None:
        return None

    column = getattr(node, "column", 0) or 0
    target = Position(line=max(0, line - 1), character=column)

    return Location(
        uri=path_to_uri(module.path),
        range=Range(start=target, end=Position(line=target.line, character=column + len(name))),
    )


def get_keyword_completions() -> CompletionList:
    """Return the default keyword completions."""
    items: List[CompletionItem] = []

    keywords = [
        # Spice keywords
        'interface', 'abstract', 'final', 'static', 'extends', 'implements',
        'data', 'enum', 'switch', 'case', 'default',
        # Python keywords
        'def', 'class', 'if', 'elif', 'else', 'for', 'while', 'return',
        'import', 'from', 'as', 'with', 'try', 'except', 'finally', 'raise',
        'pass', 'break', 'continue', 'lambda', 'and', 'or', 'not', 'in', 'is',
        'True', 'False', 'None',
    ]

    for keyword in keywords:
        items.append(CompletionItem(
            label=keyword,
            kind=CompletionItemKind.Keyword,
            detail="Spice keyword"
        ))

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
        CompletionItem(
            label="enum",
            kind=CompletionItemKind.Snippet,
            detail="Enum declaration",
            insert_text="enum ${1:Name} {\n\t${2:FIRST},\n\t${3:SECOND}\n}"
        ),
        CompletionItem(
            label="data class",
            kind=CompletionItemKind.Snippet,
            detail="Data class declaration",
            insert_text="data class ${1:Name}(${2:field}: ${3:Type});"
        ),
        CompletionItem(
            label="switch",
            kind=CompletionItemKind.Snippet,
            detail="Switch statement",
            insert_text="switch (${1:value}) {\n\tcase ${2:pattern}: {\n\t\t${3:pass};\n\t}\n\tdefault: {\n\t\t${4:pass};\n\t}\n}"
        ),
    ])

    return CompletionList(is_incomplete=False, items=items)


# How a symbol kind shows up in a completion list.
SYMBOL_COMPLETION_KINDS = {
    "class": CompletionItemKind.Class,
    "interface": CompletionItemKind.Interface,
    "function": CompletionItemKind.Function,
    "variable": CompletionItemKind.Variable,
}


def _signature_of(kind: str, symbol: object) -> str:
    """A one-line rendering of a declaration, for hover and completion detail."""
    node = getattr(symbol, "node", None)
    name = getattr(symbol, "name", "")

    if kind == "interface":
        methods = getattr(node, "methods", []) or []
        return f"interface {name} ({len(methods)} method{'s' if len(methods) != 1 else ''})"

    if kind == "class":
        parts = []
        bases = list(getattr(node, "bases", []) or [])
        interfaces = list(getattr(node, "interfaces", []) or [])
        type_parameters = [tp.name for tp in getattr(node, "type_parameters", []) or []]

        head = f"class {name}"
        if type_parameters:
            head += "<" + ", ".join(type_parameters) + ">"
        parts.append(head)
        if bases:
            parts.append("extends " + ", ".join(bases))
        if interfaces:
            parts.append("implements " + ", ".join(interfaces))
        return " ".join(parts)

    if kind in ("function", "method"):
        params = getattr(symbol, "params", []) or []
        rendered = ", ".join(
            f"{p.name}: {p.type_annotation}" if p.type_annotation else p.name
            for p in params
            if p.name != "self"
        )
        returns = getattr(symbol, "return_type", None)
        suffix = f" -> {returns}" if returns else ""
        return f"def {name}({rendered}){suffix}"

    annotation = getattr(symbol, "type_annotation", None)
    return f"{name}: {annotation}" if annotation else str(name)


def get_symbol_completions(doc: "SpiceDocument") -> List[CompletionItem]:
    """Everything the document can name - its own declarations and its imports.

    Completion used to offer keywords and nothing else, so a class from another
    module had to be typed out in full and spelled right.
    """
    if doc.spice_file is None:
        return []

    items: List[CompletionItem] = []

    for name, (kind, symbol) in sorted(workspace.visible_symbols(doc.spice_file).items()):
        items.append(CompletionItem(
            label=name,
            kind=SYMBOL_COMPLETION_KINDS.get(kind, CompletionItemKind.Variable),
            detail=_signature_of(kind, symbol),
            documentation=MarkupContent(
                kind=MarkupKind.Markdown,
                value=f"```spice\n{_signature_of(kind, symbol)}\n```",
            ),
            # After keywords, which are what a bare prefix usually means.
            sort_text=f"zz{name}",
        ))

    return items


def _annotation_doc(name: str) -> str:
    """Build hover/detail markdown for a registered compile-time annotation."""
    proc = all_processors().get(name)
    if proc is None:
        return ""
    doc = (type(proc).__doc__ or "").strip()
    targets = getattr(proc, "targets", ()) or ()
    target_str = ", ".join(t.__name__ for t in targets) if targets else "any declaration"
    parts = [f"**@!{name}** compile-time annotation", f"Applies to: {target_str}"]
    if doc:
        parts.append(doc)
    return "\n\n".join(parts)


def detect_annotation_context(source: str, position: Position) -> Dict:
    """Detect whether the cursor is typing an annotation prefix.

    Returns {"in_annotation": bool, "retention": "compile_time"|"runtime", "partial": str}.
    """
    result = {"in_annotation": False, "retention": "runtime", "partial": ""}

    lines = source.split('\n')
    if position.line >= len(lines):
        return result

    line_before_cursor = lines[position.line][:position.character]

    match = re.match(r'^\s*(@!?)([\w.]*)$', line_before_cursor)
    if match:
        result["in_annotation"] = True
        result["retention"] = "compile_time" if match.group(1) == "@!" else "runtime"
        result["partial"] = match.group(2)
    return result


def get_annotation_completions(context: Dict) -> CompletionList:
    """Suggest registered compile-time annotation processors for '@!'."""
    items: List[CompletionItem] = []

    # Only '@!' has a known registry; runtime '@' decorators are arbitrary Python.
    if context["retention"] == "compile_time":
        partial = context["partial"].lower()
        for name in sorted(all_processors()):
            if partial and not name.lower().startswith(partial):
                continue
            items.append(CompletionItem(
                label=name,
                kind=CompletionItemKind.Function,
                detail="Compile-time annotation",
                documentation=MarkupContent(kind=MarkupKind.Markdown, value=_annotation_doc(name)),
            ))

    return CompletionList(is_incomplete=False, items=items)


@server.feature("initialize")
def initialize(ls: LanguageServer, params: InitializeParams):
    """Take the client's workspace folders and any configured search paths.

    The folders matter: they tell the server what a dotted module name resolves
    against. Guessing that from the open file alone works for a flat directory
    and gets a package layout wrong.
    """
    folders: List[Path] = []

    for folder in getattr(params, "workspace_folders", None) or []:
        path = uri_to_path(getattr(folder, "uri", ""))
        if path is not None:
            folders.append(path)

    root_uri = getattr(params, "root_uri", None)
    if not folders and root_uri:
        path = uri_to_path(root_uri)
        if path is not None:
            folders.append(path)

    extra = _configured_lookup_paths(getattr(params, "initialization_options", None))

    logger.info(f"Workspace folders: {folders}; extra lookup paths: {extra}")
    workspace.configure(workspace_folders=folders, extra_paths=extra)


def _configured_lookup_paths(options: object) -> List[Path]:
    """`spice.lookupPaths` from the client, if it sent any.

    The server used to carry a note wishing for this - a project that keeps its
    sources under `src/` has no way to say so otherwise.
    """
    if not options:
        return []

    raw = None
    if isinstance(options, dict):
        raw = options.get("lookupPaths")
    else:
        raw = getattr(options, "lookupPaths", None)

    if not raw:
        return []

    paths: List[Path] = []
    for entry in raw:
        try:
            candidate = Path(str(entry)).expanduser()
            if candidate.exists():
                paths.append(candidate)
            else:
                logger.warning(f"Configured lookup path does not exist: {candidate}")
        except Exception as error:
            logger.warning(f"Bad lookup path {entry!r}: {error}")

    return paths


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
    """Handle document save event.

    A save is the moment what other files see of this one changes, so the
    cached projects go and every open document is re-checked against fresh
    sources.
    """
    uri = params.text_document.uri
    logger.info(f"Document saved: {uri}")

    invalidate_caches()
    _republish_all(ls)


def _handle_import_definition(uri: str, import_ctx: Dict) -> Optional[Location]:
    """Handle go-to-definition for import statements."""
    module_name = import_ctx["module"]
    symbol_name = import_ctx.get("name")

    # Resolve the module to a file path
    module_path = resolve_module_path(module_name, uri)
    if not module_path:
        logger.debug(f"Could not resolve module: {module_name}")
        return None

    logger.debug(f"Resolved module {module_name} to {module_path}")

    target_uri = path_to_uri(module_path)

    if import_ctx["type"] == "module":
        # Jump to the module file (line 0)
        return Location(
            uri=target_uri,
            range=Range(
                start=Position(line=0, character=0),
                end=Position(line=0, character=0)
            )
        )

    elif import_ctx["type"] == "name" and symbol_name:
        # Jump to the specific symbol in the module
        symbol_pos = find_symbol_in_file(module_path, symbol_name)
        if symbol_pos:
            line, col = symbol_pos
            return Location(
                uri=target_uri,
                range=Range(
                    start=Position(line=line, character=col),
                    end=Position(line=line, character=col + len(symbol_name))
                )
            )
        else:
            # Symbol not found, just jump to file
            logger.debug(f"Symbol {symbol_name} not found in {module_path}, jumping to file")
            return Location(
                uri=target_uri,
                range=Range(
                    start=Position(line=0, character=0),
                    end=Position(line=0, character=0)
                )
            )

    return None


@server.feature("textDocument/completion")
def completions(params: CompletionParams) -> Optional[CompletionList]:
    """Provide completion items."""
    uri = params.text_document.uri
    position = params.position

    if uri not in documents:
        return None

    doc = documents[uri]

    # Check if we're typing an annotation prefix '@' / '@!'
    annotation_ctx = detect_annotation_context(doc.source, position)
    if annotation_ctx["in_annotation"]:
        logger.debug(f"Annotation context detected: {annotation_ctx}")
        return get_annotation_completions(annotation_ctx)

    # Check if we're in an import context
    import_ctx = detect_import_context(doc.source, position)

    if import_ctx["in_import"]:
        logger.debug(f"Import context detected: {import_ctx}")
        return get_import_completions(uri, import_ctx)

    # Keywords and snippets, plus everything this file can actually name.
    result = get_keyword_completions()
    result.items.extend(get_symbol_completions(doc))
    return result


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
    col = position.character

    # Check for scientific notation number at cursor position
    # Pattern matches: 1e10, 2.5e-3, 3E+5, 4.0E2, etc.
    sci_pattern = re.compile(r'\d+\.?\d*[eE][+-]?\d+')
    for match in sci_pattern.finditer(line):
        if match.start() <= col <= match.end():
            sci_str = match.group(0)
            try:
                # Use Decimal for arbitrary precision (no floating point errors)
                parsed_value = Decimal(sci_str)
                abs_value = abs(parsed_value)
                # Keep scientific notation only for extreme values
                if abs_value >= Decimal('1e30') or (abs_value < Decimal('1e-30') and abs_value != 0):
                    formatted = f"{parsed_value:.10g}"
                else:
                    # Convert to string, strip trailing zeros, add comma separators
                    raw = format(parsed_value, 'f')
                    if '.' in raw:
                        int_part, dec_part = raw.split('.')
                        dec_part = dec_part.rstrip('0')
                        if dec_part:
                            formatted = f"{int(int_part):,}.{dec_part}"
                        else:
                            formatted = f"{int(int_part):,}"
                    else:
                        formatted = f"{int(raw):,}"
                return Hover(
                    contents=MarkupContent(
                        kind=MarkupKind.Markdown,
                        value=f"**Scientific Notation for:** `{formatted}`"
                    ),
                    range=Range(
                        start=Position(line=position.line, character=match.start()),
                        end=Position(line=position.line, character=match.end())
                    )
                )
            except (ValueError, InvalidOperation):
                pass

    # Simple word extraction for keywords/identifiers
    start = col
    end = col

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
        'data': '**data** modifier\n\nDeclares a data class (auto-generated `__init__`, equality, and repr from its fields)',
        'enum': '**enum** keyword\n\nDeclares an enumeration of named members',
        'switch': '**switch** keyword\n\nMatches a value against `case` patterns, with an optional `default` branch',
        'case': '**case** keyword\n\nA branch of a `switch` statement matched against a pattern',
        'default': '**default** keyword\n\nThe fallback branch of a `switch` statement',
    }

    if word in hover_docs:
        return Hover(
            contents=MarkupContent(
                kind=MarkupKind.Markdown,
                value=hover_docs[word]
            )
        )

    if word in all_processors():
        return Hover(
            contents=MarkupContent(
                kind=MarkupKind.Markdown,
                value=_annotation_doc(word)
            )
        )

    # A declared name: show what it actually is, and where it came from.
    if doc.spice_file is not None and word:
        found = workspace.declaration_of(doc.spice_file, word)
        if found is not None:
            module, kind, symbol = found
            lines = [f"```spice\n{_signature_of(kind, symbol)}\n```"]
            if module.path != doc.spice_file.path:
                lines.append(f"*from `{module.path.name}`*")
            return Hover(
                contents=MarkupContent(kind=MarkupKind.Markdown, value="\n\n".join(lines)),
                range=Range(
                    start=Position(line=position.line, character=start),
                    end=Position(line=position.line, character=end),
                ),
            )

    return None


@server.feature("textDocument/definition")
def definition(params: DefinitionParams) -> Optional[Location]:
    """Provide go-to-definition support."""
    uri = params.text_document.uri
    position = params.position

    if uri not in documents:
        return None

    doc = documents[uri]

    # Check if cursor is on an import statement first
    import_ctx = detect_import_definition_context(doc.source, position)
    if import_ctx:
        logger.debug(f"Import definition context: {import_ctx}")
        return _handle_import_definition(uri, import_ctx)

    # Get word at position
    lines = doc.source.split('\n')
    if position.line >= len(lines):
        return None

    line = lines[position.line]

    # Extract word at cursor position
    start = position.character
    end = position.character

    while start > 0 and (line[start - 1].isalnum() or line[start - 1] == '_'):
        start -= 1

    while end < len(line) and (line[end].isalnum() or line[end] == '_'):
        end += 1

    word = line[start:end]

    if not word:
        return None

    logger.debug(f"Looking up definition for: '{word}'")

    # Searched through the project, so a name that came in on an import
    # resolves - and resolves to the module that actually declares it.
    if doc.spice_file is not None:
        found = workspace.declaration_of(doc.spice_file, word)
        if found is not None:
            module, kind, symbol = found
            node = symbol.node
            logger.debug(f"Found {kind} '{word}' in {module.path}")
            location = _declaration_location(module, node, word)
            if location is not None:
                return location

    logger.debug(f"Symbol '{word}' not found")
    return None


@server.feature("textDocument/documentSymbol")
def document_symbols(params: DocumentSymbolParams) -> Optional[List[DocumentSymbol]]:
    """The outline: what this file declares, with its members nested inside."""
    uri = params.text_document.uri
    doc = documents.get(uri)
    if doc is None or doc.ast is None:
        return None

    symbols: List[DocumentSymbol] = []
    for node in doc.ast.body:
        symbol = _outline_symbol(node)
        if symbol is not None:
            symbols.append(symbol)

    return symbols or None


# What each declaration looks like in the outline.
OUTLINE_KINDS = {
    ClassDeclaration: SymbolKind.Class,
    DataClassDeclaration: SymbolKind.Struct,
    EnumDeclaration: SymbolKind.Enum,
    InterfaceDeclaration: SymbolKind.Interface,
    FunctionDeclaration: SymbolKind.Function,
}


def _node_range(node: object, name: str) -> Range:
    """A node's own line, which is all the AST records of its extent.

    Declarations carry a start position and no end, so the whole declaration and
    its selection range are the same span. Editors accept that; it just means
    the outline highlights the header rather than the body.
    """
    line = max(0, (getattr(node, "line", 1) or 1) - 1)
    column = getattr(node, "column", 0) or 0
    return Range(
        start=Position(line=line, character=column),
        end=Position(line=line, character=column + len(name)),
    )


def _outline_symbol(node: object) -> Optional[DocumentSymbol]:
    kind = OUTLINE_KINDS.get(type(node))
    if kind is None:
        return None

    name = getattr(node, "name", None)
    if not name:
        return None

    children: List[DocumentSymbol] = []
    for member in getattr(node, "body", []) or []:
        child = _outline_symbol(member)
        if child is not None:
            children.append(child)

    # An interface's signatures aren't in `body`, and a data class's fields
    # aren't either - both are worth showing.
    for signature in getattr(node, "methods", []) or []:
        signature_name = getattr(signature, "name", None)
        if signature_name:
            children.append(DocumentSymbol(
                name=signature_name,
                kind=SymbolKind.Method,
                range=_node_range(signature, signature_name),
                selection_range=_node_range(signature, signature_name),
            ))

    for field in getattr(node, "fields", []) or []:
        field_name = getattr(field, "name", None)
        if field_name:
            children.append(DocumentSymbol(
                name=field_name,
                detail=getattr(field, "type_annotation", None),
                kind=SymbolKind.Field,
                range=_node_range(node, field_name),
                selection_range=_node_range(node, field_name),
            ))

    span = _node_range(node, name)
    return DocumentSymbol(
        name=name,
        kind=kind,
        range=span,
        selection_range=span,
        children=children or None,
    )


@server.feature("workspace/didChangeWatchedFiles")
def did_change_watched_files(ls: LanguageServer, params: DidChangeWatchedFilesParams):
    """A .spc file changed outside the editor - drop what was read off disk."""
    logger.info("Watched files changed; invalidating workspace")
    invalidate_caches()
    _republish_all(ls)


def _republish_all(ls: LanguageServer) -> None:
    """Re-check every open document and publish the result.

    A change in one module can turn a diagnostic in another on or off - that is
    the point of resolving across modules - so the file being edited is not the
    only one whose squiggles may be out of date.
    """
    for uri, existing in list(documents.items()):
        refreshed = SpiceDocument(uri, existing.source)
        documents[uri] = refreshed
        ls.text_document_publish_diagnostics(
            PublishDiagnosticsParams(uri=uri, diagnostics=refreshed.diagnostics)
        )


def start_server():
    """Start the Spice LSP server."""
    logger.info("Starting Spice Language Server")
    server.start_io()


if __name__ == "__main__":
    start_server()

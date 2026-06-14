"""
Spice Language Server Protocol Implementation.

This LSP server integrates with the Spice compiler's Lexer, Parser, and Type system
to provide rich IDE features for .spc files.
"""

import logging
import sys
import re
import sysconfig
import site
import importlib
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from urllib.parse import urlparse, unquote

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
)

from spice.lexer import Lexer, TokenType
from spice.parser import Parser
from spice.parser.ast_nodes import (
    ClassDeclaration,
    FunctionDeclaration,
    InterfaceDeclaration,
    DataClassDeclaration,
    EnumDeclaration,
)
from spice.compilation.spicefile import SpiceFile
from spice.compilation.checks import SymbolTableBuilder, TypeChecker, MethodOverloadResolver, InterfaceChecker, FinalChecker, CheckError
from spice.errors import SpiceError

import spice.annotations.builtins
from spice.annotations import all_processors

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
        self.spice_file: Optional[SpiceFile] = None
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

            if self.ast:
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
            self.spice_file = SpiceFile.empty(self.source)
            self.spice_file.ast = self.ast

            logger.debug("Building symbol table...")
            symbol_builder = SymbolTableBuilder()
            symbol_builder.check(self.spice_file)
            logger.debug(f"Symbol table built: {self.spice_file.symbol_table is not None}")
            if self.spice_file.symbol_table:
                classes = list(self.spice_file.symbol_table.classes.keys())
                interfaces = list(self.spice_file.symbol_table.interfaces.keys())
                logger.debug(f"Symbol table - classes: {classes}, interfaces: {interfaces}")

            logger.debug("Checking method overloads...")
            overload_resolver = MethodOverloadResolver()
            if not overload_resolver.check(self.spice_file):
                for error in overload_resolver.errors:
                    self.diagnostics.append(self._make_diagnostic(error, "spice-overload"))

            logger.debug("Running type checker...")
            type_checker = TypeChecker()
            if not type_checker.check(self.spice_file):
                for error in type_checker.errors:
                    self.diagnostics.append(self._make_diagnostic(error, "spice-type"))

            logger.debug("Checking interface implementations...")
            interface_checker = InterfaceChecker()
            if not interface_checker.check(self.spice_file):
                for error in interface_checker.errors:
                    self.diagnostics.append(self._make_diagnostic(error, "spice-interface"))

            logger.debug("Checking final constraints...")
            final_checker = FinalChecker()
            if not final_checker.check(self.spice_file):
                for error in final_checker.errors:
                    self.diagnostics.append(self._make_diagnostic(error, "spice-final"))

            logger.debug(f"Semantic checks complete, {len(self.diagnostics)} diagnostics")

        except Exception as e:
            logger.exception(f"Error during semantic checks: {e}")


# Document cache
documents: dict[str, SpiceDocument] = {}

# Module discovery caches
_module_cache: Dict[str, Dict] = {}  # uri -> {stdlib, packages, spice, all}
_export_cache: Dict[str, List[Tuple[str, CompletionItemKind]]] = {}  # module_name -> [(name, kind)]


# Import Intellisense helpers
def _uri_to_path(uri: str) -> Optional[Path]:
    """Convert a file:// URI to a Path."""
    if uri.startswith("file://"):
        # Handle Windows paths (file:///C:/...)
        path_str = unquote(urlparse(uri).path)
        if sys.platform == "win32" and path_str.startswith("/"):
            path_str = path_str[1:]  # Remove leading slash for Windows
        return Path(path_str)
    return None


def get_lookup_paths(document_uri: str) -> List[Path]:
    """Get lookup paths for module discovery, similar to pipeline.py."""
    paths: List[Path] = []

    # I don't know if the order is *really* important, should also make a vscode config for more paths tbf (and ofc cli commands)

    # Document's directory
    doc_path = _uri_to_path(document_uri)
    if doc_path and doc_path.parent.exists():
        paths.append(doc_path.parent)

    # Current working directory
    cwd = Path.cwd()
    if cwd.exists() and cwd not in paths:
        paths.append(cwd)

    # Python environment paths
    try:
        purelib = sysconfig.get_path('purelib')
        if purelib:
            paths.append(Path(purelib))
    except Exception:
        pass

    try:
        platlib = sysconfig.get_path('platlib')
        if platlib:
            paths.append(Path(platlib))
    except Exception:
        pass

    try:
        user_site = site.getusersitepackages()
        if user_site:
            paths.append(Path(user_site))
    except Exception:
        pass

    try:
        stdlib = sysconfig.get_path('stdlib')
        if stdlib:
            paths.append(Path(stdlib))
    except Exception:
        pass

    try:
        for site_path in site.getsitepackages():
            paths.append(Path(site_path))
    except Exception:
        pass

    # Filter to existing, unique paths
    seen = set()
    result = []
    for p in paths:
        if p and p.exists():
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                result.append(p)

    return result


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
    """Resolve a module name to its file path."""
    doc_path = _uri_to_path(document_uri)
    paths = get_lookup_paths(document_uri)

    # Convert dotted module name to path
    module_path_part = module_name.replace('.', '/')

    for base_path in paths:
        # Check for .spc file
        spc_file = base_path / f"{module_path_part}.spc"
        if spc_file.exists():
            return spc_file

        # Check for .spc package
        spc_pkg = base_path / module_path_part / "__init__.spc"
        if spc_pkg.exists():
            return spc_pkg

        # Check for .py file
        py_file = base_path / f"{module_path_part}.py"
        if py_file.exists():
            return py_file

        # Check for Python package
        py_pkg = base_path / module_path_part / "__init__.py"
        if py_pkg.exists():
            return py_pkg

    return None


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


def path_to_uri(path: Path) -> str:
    """Convert a Path to a file:// URI."""
    resolved = path.resolve()
    if sys.platform == "win32":
        # Windows: file:///C:/path/to/file
        return f"file:///{resolved.as_posix()}"
    else:
        # Unix: file:///path/to/file
        return f"file://{resolved.as_posix()}"


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

    # Default: return keyword completions
    return get_keyword_completions()


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

    # Check SymbolTable for interfaces, classes, and functions
    if doc.spice_file and doc.spice_file.symbol_table:
        symbol_table = doc.spice_file.symbol_table

        # Look up interface
        if word in symbol_table.interfaces:
            interface_symbol = symbol_table.interfaces[word]
            node = interface_symbol.node
            logger.debug(f"Found interface: {word} at line {node.line}, column {node.column}")
            return Location(
                uri=uri,
                range=Range(
                    start=Position(line=node.line - 1, character=node.column),
                    end=Position(line=node.line - 1, character=node.column + len(word))
                )
            )

        # Look up class
        if word in symbol_table.classes:
            class_symbol = symbol_table.classes[word]
            node = class_symbol.node
            logger.debug(f"Found class: {word} at line {node.line}, column {node.column}")
            return Location(
                uri=uri,
                range=Range(
                    start=Position(line=node.line - 1, character=node.column),
                    end=Position(line=node.line - 1, character=node.column + len(word))
                )
            )

        # Look up global function
        global_scope = symbol_table.scopes.get("global")
        if global_scope and word in global_scope.functions:
            func_symbols = global_scope.functions[word]
            if func_symbols:
                node = func_symbols[0].node
                logger.debug(f"Found function: {word} at line {node.line}, column {node.column}")
                return Location(
                    uri=uri,
                    range=Range(
                        start=Position(line=node.line - 1, character=node.column),
                        end=Position(line=node.line - 1, character=node.column + len(word))
                    )
                )

        # Look up method in any class
        for class_name, class_symbol in symbol_table.classes.items():
            if word in class_symbol.methods:
                method_symbols = class_symbol.methods[word]
                if method_symbols:
                    node = method_symbols[0].node
                    logger.debug(f"Found method: {class_name}.{word} at line {node.line}, column {node.column}")
                    return Location(
                        uri=uri,
                        range=Range(
                            start=Position(line=node.line - 1, character=node.column),
                            end=Position(line=node.line - 1, character=node.column + len(word))
                        )
                    )

    logger.debug(f"Symbol '{word}' not found")
    return None


def start_server():
    """Start the Spice LSP server."""
    logger.info("Starting Spice Language Server")
    server.start_io()


if __name__ == "__main__":
    start_server()

"""The workspace behind the language server: one SpiceProject per root.

The server used to resolve modules itself - its own lookup-path list, its own
`module name -> file` search, its own regex reading of import lines - alongside
the compiler's. Two implementations of the same thing drift, and the editor's
was the one that got it wrong: it never saw past the file being edited, so a
class implementing an interface declared in another module was underlined as
implementing an unknown one, in code that compiles.

This is the other half of that fix. A SpiceProject holds the module graph, and
the server asks it the same questions the compiler does.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote

try:
    from spice.compilation import BuildFlags, SpiceProject
except ImportError as error:  # pragma: no cover - depends on what is installed
    # SpiceProject is what makes resolving across modules possible, and it
    # arrived after 0.3.1. Without this the failure is a bare ImportError deep
    # in a server the editor started in the background, which nobody sees.
    raise ImportError(
        "This version of spice-lsp needs a newer spice-lang than the one "
        "installed (it requires spice.compilation.SpiceProject). "
        "Upgrade with: pip install -U spice-lang"
    ) from error

from spice.compilation.pipeline import SpicePipeline
from spice.compilation.spicefile import SpiceFile
from spice.printils import set_log_mode
from spice.compilation.checks import (
    FinalChecker,
    GenericBoundChecker,
    InterfaceChecker,
    MethodOverloadResolver,
    SymbolTableBuilder,
    TypeChecker,
)

logger = logging.getLogger(__name__)

# The compiler's stage logging goes to stdout, and stdout is the LSP's
# JSON-RPC channel - a single "Tokenizing file: ..." line corrupts the stream
# and the client drops the connection. Silence it before anything can run.
set_log_mode("none")


# Files that mark the top of a project. A root is what module names resolve
# against, so guessing it wrong makes every dotted import unresolvable.
ROOT_MARKERS = (
    "build.spice.py",
    "pyproject.toml",
    "setup.py",
    ".git",
    "__main__.spc",
)

# How far up from a file to look for one of those before giving up and calling
# the file's own directory the root.
MAX_ROOT_DEPTH = 24


def uri_to_path(uri: str) -> Optional[Path]:
    """A `file://` URI as a Path, or None for anything else."""
    if not uri.startswith("file://"):
        return None

    path_str = unquote(urlparse(uri).path)
    # file:///C:/x on Windows arrives with a leading slash the drive can't take.
    if sys.platform == "win32" and path_str.startswith("/"):
        path_str = path_str[1:]
    return Path(path_str)


def path_to_uri(path: Path) -> str:
    """A Path as a `file://` URI."""
    resolved = Path(path).resolve()
    if sys.platform == "win32":
        return f"file:///{resolved.as_posix()}"
    return f"file://{resolved.as_posix()}"


@dataclass
class Analysis:
    """What one pass over an open document produced."""

    file: SpiceFile
    project: SpiceProject
    # (source-label, error) - the label becomes the diagnostic's `source`, so
    # the editor can say which check objected.
    errors: List[Tuple[str, Any]] = field(default_factory=list)
    warnings: List[Tuple[str, Any]] = field(default_factory=list)

    def add(self, source: str, errors: Any) -> None:
        for error in errors or []:
            self.errors.append((source, error))


class SpiceWorkspace:
    """Every project the editor has a file open in.

    Projects are cached because building one re-reads and re-parses the whole
    import graph, which is far too much work to repeat on a keystroke. The cache
    is dropped wholesale when a file is saved or changes on disk - imprecise,
    but a rebuild is cheap next to getting a stale answer, and it keeps the
    invalidation story small enough to be obviously correct.
    """

    def __init__(self) -> None:
        self._projects: Dict[Path, SpiceProject] = {}
        self._roots: Dict[Path, Path] = {}
        self.workspace_folders: List[Path] = []
        self.extra_paths: List[Path] = []

    # Configuration #

    def configure(self, workspace_folders: List[Path] = None, extra_paths: List[Path] = None) -> None:
        """Take the client's workspace folders and any extra search paths."""
        if workspace_folders is not None:
            self.workspace_folders = [Path(folder) for folder in workspace_folders]
        if extra_paths is not None:
            self.extra_paths = [Path(p) for p in extra_paths]
        self.invalidate()

    # Roots and projects #

    def root_for(self, path: Path) -> Path:
        """The directory `path`'s module names resolve against.

        A workspace folder containing the file wins - the client knows what the
        user opened better than any heuristic. Failing that, walk up looking for
        something that marks the top of a project, and settle for the file's own
        directory.
        """
        path = Path(path).resolve()
        cached = self._roots.get(path)
        if cached is not None:
            return cached

        root = self._detect_root(path)
        self._roots[path] = root
        return root

    def _detect_root(self, path: Path) -> Path:
        # Deepest containing workspace folder, so nested folders behave.
        containing = [
            folder.resolve()
            for folder in self.workspace_folders
            if self._contains(folder.resolve(), path)
        ]
        if containing:
            return max(containing, key=lambda folder: len(folder.parts))

        current = path.parent
        for _ in range(MAX_ROOT_DEPTH):
            for marker in ROOT_MARKERS:
                candidate = current / marker
                # __main__.spc marks the root it sits in, but the file being
                # edited may *be* it - that still makes its directory the root.
                if candidate.exists():
                    return current
            if current.parent == current:
                break
            current = current.parent

        return path.parent

    @staticmethod
    def _contains(folder: Path, path: Path) -> bool:
        try:
            path.relative_to(folder)
            return True
        except ValueError:
            return False

    def flags_for(self, path: Path) -> BuildFlags:
        """Build flags for an editor pass: analyze, never write, stay quiet."""
        return BuildFlags(source=Path(path), output=None, emit="py", log_mode="none")

    def project_for(self, path: Path) -> SpiceProject:
        """The project `path` belongs to, built once and kept."""
        root = self.root_for(path)
        project = self._projects.get(root)
        if project is not None:
            return project

        logger.debug(f"Creating project rooted at {root}")
        project = SpiceProject(root, self.flags_for(path))
        for extra in self.extra_paths:
            project.add_lookup_path(extra)
        self._projects[root] = project
        return project

    def invalidate(self, path: Optional[Path] = None) -> None:
        """Forget cached projects, so the next pass re-reads from disk.

        Called on save and on a watched file changing. Everything goes rather
        than just the file that changed: what a module exports is reachable from
        anything that imports it, transitively, and tracking that precisely is
        the incremental design this does not yet have.
        """
        self._projects.clear()
        self._roots.clear()

    # Analysis #

    def analyze(self, path: Path, source: str) -> Analysis:
        """Re-check one open document against its project.

        The buffer wins over what is on disk - that is the whole point of doing
        this in an editor - so the file is reloaded from `source` and everything
        derived from the old text is dropped.
        """
        path = Path(path).resolve()
        project = self.project_for(path)
        flags = self.flags_for(path)

        file = project.module_for(path)
        file.reload(source)
        project.entry = file

        result = Analysis(file=file, project=project)

        SpicePipeline.tokenize(file, flags)
        SpicePipeline.parse(file, flags)

        try:
            SpicePipeline.resolve_imports(file, project, flags)
        except Exception as error:
            # An unresolved import is worth saying out loud, but it must not
            # stop the rest of the file being checked - half the time it is a
            # module the author has not written yet.
            result.errors.append(("spice-import", error))

        self._build_dependency_symbols(project, file, flags)
        self._check(file, result)
        return result

    def _build_dependency_symbols(self, project: SpiceProject, file: SpiceFile, flags: BuildFlags) -> None:
        """Parse everything `file` imports and build its symbol table.

        Only the symbol tables - the imported modules are not being compiled,
        they are being read so the checks on the edited file can resolve a name
        that crosses a module boundary.
        """
        seen = {file.path.resolve()}
        frontier = list(file.spc_imports)

        while frontier:
            module = frontier.pop()
            here = module.path.resolve()
            if here in seen:
                continue
            seen.add(here)

            try:
                if not module.tokens:
                    SpicePipeline.tokenize(module, flags)
                    SpicePipeline.parse(module, flags)
                    SpicePipeline.resolve_imports(module, project, flags)
                if module.symbol_table is None:
                    SymbolTableBuilder().check(module)
            except Exception as error:
                # A dependency that will not parse is that file's problem, and
                # is reported when it is the one being edited. Here it just
                # means some names stay unresolved.
                logger.debug(f"Skipping unreadable dependency {module.path}: {error}")
                continue

            frontier.extend(module.spc_imports)

    def _check(self, file: SpiceFile, result: Analysis) -> None:
        """Run the compiler's checks over the edited file.

        The same set the pipeline runs, minus the annotation stage: that one
        rewrites the tree, and every position the editor reports afterwards
        would refer to code the author never wrote.
        """
        SymbolTableBuilder().check(file)

        checks = (
            (MethodOverloadResolver(), "spice-overload"),
            (TypeChecker(), "spice-type"),
            (InterfaceChecker(), "spice-interface"),
            (GenericBoundChecker(), "spice-generic"),
            (FinalChecker(), "spice-final"),
        )

        for check, label in checks:
            try:
                if not check.check(file):
                    result.add(label, check.errors)
            except Exception as error:
                logger.exception(f"{label} failed on {file.path}: {error}")

        for warning in getattr(file, "warnings", []):
            result.warnings.append(("spice", warning))

    # Queries #

    def declaration_of(self, file: SpiceFile, name: str) -> Optional[Tuple[SpiceFile, str, Any]]:
        """Where `name` is declared, as seen from `file`.

        `(module, kind, symbol)` or None. The module matters: without it a jump
        to an imported class lands on the current file at the imported class's
        line number, which is a different piece of code entirely.
        """
        project = getattr(file, "project", None)
        if project is None:
            return None
        return project.find_declaration(file, name)

    def visible_symbols(self, file: SpiceFile) -> Dict[str, Tuple[str, Any]]:
        """Every name `file` can use, mapped to `(kind, symbol)`.

        Its own declarations plus whatever its imports expose, which is what a
        completion list should be offering.
        """
        project = getattr(file, "project", None)
        if project is None:
            table = getattr(file, "symbol_table", None)
            if table is None:
                return {}
            return self._symbols_from_table(table)

        found: Dict[str, Tuple[str, Any]] = {}
        # Farthest first, so a nearer declaration overwrites it.
        for level in reversed(project.import_levels(file)):
            for module in level:
                table = getattr(module, "symbol_table", None)
                if table is not None:
                    found.update(self._symbols_from_table(table))
        return found

    @staticmethod
    def _symbols_from_table(table: Any) -> Dict[str, Tuple[str, Any]]:
        found: Dict[str, Tuple[str, Any]] = {}

        for name, symbol in table.interfaces.items():
            found[name] = ("interface", symbol)
        for name, symbol in table.classes.items():
            found[name] = ("class", symbol)

        global_scope = table.scopes.get("global")
        if global_scope is not None:
            for name, overloads in global_scope.functions.items():
                if overloads:
                    found[name] = ("function", overloads[0])
            for name, variable in global_scope.variables.items():
                found[name] = ("variable", variable)

        return found

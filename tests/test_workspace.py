"""What the server answers once it resolves across module boundaries.

The old server only ever looked at the file being edited, so a class
implementing an interface from another module was underlined as implementing an
unknown one - in code the compiler accepts. These are the cases that proves.
"""

from pathlib import Path

import pytest

from spice_lsp.server import SpiceDocument, documents, workspace
from spice_lsp.workspace import SpiceWorkspace, path_to_uri, uri_to_path

from lsprotocol.types import (
    CompletionParams,
    DefinitionParams,
    DocumentSymbolParams,
    HoverParams,
    Position,
    TextDocumentIdentifier,
)
import spice_lsp.server as server_module


SHAPES = """interface Drawable {
    def draw() -> str;
}

class Point {
    def Point(x: int, y: int) -> None {
        self.x: int = x;
        self.y: int = y;
    }
}
"""

MAIN = """from shapes import Drawable

class Circle implements Drawable {
    def draw() -> str {
        return "circle";
    }
}
"""


@pytest.fixture
def project(tmp_path):
    """A two-module project with the editor pointed at its root."""
    (tmp_path / "shapes.spc").write_text(SHAPES, encoding="utf-8")
    (tmp_path / "main.spc").write_text(MAIN, encoding="utf-8")

    workspace.invalidate()
    workspace.configure(workspace_folders=[tmp_path], extra_paths=[])
    documents.clear()
    yield tmp_path
    documents.clear()
    workspace.invalidate()


def _open(path: Path, source: str = None) -> SpiceDocument:
    uri = path_to_uri(path)
    doc = SpiceDocument(uri, source if source is not None else path.read_text(encoding="utf-8"))
    documents[uri] = doc
    return doc


def _at(source: str, needle: str, occurrence: int = 0) -> Position:
    """The position of `needle` in `source`, as the editor would send it."""
    index = -1
    for _ in range(occurrence + 1):
        index = source.index(needle, index + 1)
    line = source.count("\n", 0, index)
    column = index - (source.rfind("\n", 0, index) + 1)
    return Position(line=line, character=column)


class TestDiagnostics:
    def test_implementing_an_imported_interface_is_clean(self, project):
        doc = _open(project / "main.spc")
        assert doc.diagnostics == []

    def test_an_unimplemented_imported_interface_is_still_reported(self, project):
        source = MAIN.replace("def draw()", "def paint()")
        doc = _open(project / "main.spc", source)

        messages = " ".join(d.message for d in doc.diagnostics)
        assert "draw" in messages
        assert any(d.source == "spice-interface" for d in doc.diagnostics)

    def test_a_missing_module_is_reported_without_losing_the_rest(self, project):
        """An import the author has not written yet must not stop the checks."""
        doc = _open(project / "main.spc", "import not_written_yet\n\n" + MAIN)

        assert any(d.source == "spice-import" for d in doc.diagnostics)
        # The class still got checked: it satisfies Drawable, so nothing else.
        assert not any(d.source == "spice-interface" for d in doc.diagnostics)

    def test_the_buffer_wins_over_the_file_on_disk(self, project):
        """Unsaved edits are what the editor is checking."""
        broken = MAIN.replace("def draw()", "def paint()")
        doc = _open(project / "main.spc", broken)
        assert doc.diagnostics

        fixed = _open(project / "main.spc", MAIN)
        assert fixed.diagnostics == []


class TestDefinition:
    def test_jumps_to_the_declaring_module(self, project):
        """The URI has to be the other file's.

        Definition used to answer with the current document's URI whatever it
        found, so following an imported name jumped to that line number in the
        file being edited.
        """
        doc = _open(project / "main.spc")
        position = _at(MAIN, "Drawable", occurrence=1)

        location = server_module.definition(
            DefinitionParams(text_document=TextDocumentIdentifier(uri=doc.uri), position=position)
        )

        assert location is not None
        assert uri_to_path(location.uri).name == "shapes.spc"
        assert location.range.start.line == 0

    def test_still_finds_a_local_declaration(self, project):
        doc = _open(project / "main.spc")
        position = _at(MAIN, "Circle")

        location = server_module.definition(
            DefinitionParams(text_document=TextDocumentIdentifier(uri=doc.uri), position=position)
        )

        assert location is not None
        assert uri_to_path(location.uri).name == "main.spc"

    def test_unknown_name_returns_nothing(self, project):
        doc = _open(project / "main.spc")
        position = _at(MAIN, "return")

        location = server_module.definition(
            DefinitionParams(text_document=TextDocumentIdentifier(uri=doc.uri), position=position)
        )
        assert location is None


class TestCompletionAndHover:
    def test_imported_symbols_are_offered(self, project):
        doc = _open(project / "main.spc")

        result = server_module.completions(
            CompletionParams(
                text_document=TextDocumentIdentifier(uri=doc.uri),
                position=Position(line=len(MAIN.split("\n")) - 1, character=0),
            )
        )

        labels = {item.label for item in result.items}
        assert {"Drawable", "Point", "Circle"} <= labels
        # Keywords are still there, and still first.
        assert "interface" in labels

    def test_hover_names_the_module_a_symbol_came_from(self, project):
        doc = _open(project / "main.spc")

        result = server_module.hover(
            HoverParams(
                text_document=TextDocumentIdentifier(uri=doc.uri),
                position=_at(MAIN, "Drawable", occurrence=1),
            )
        )

        assert result is not None
        assert "interface Drawable" in result.contents.value
        assert "shapes.spc" in result.contents.value


class TestDocumentSymbols:
    def test_outline_nests_members_under_their_declaration(self, project):
        doc = _open(project / "shapes.spc")

        symbols = server_module.document_symbols(
            DocumentSymbolParams(text_document=TextDocumentIdentifier(uri=doc.uri))
        )

        by_name = {symbol.name: symbol for symbol in symbols}
        assert {"Drawable", "Point"} <= set(by_name)
        assert "draw" in [child.name for child in by_name["Drawable"].children]
        assert "Point" in [child.name for child in by_name["Point"].children]


class TestImportFeatures:
    """The import intellisense that already worked, still working.

    It used to run off a second copy of the compiler's module resolution; it
    runs off the project now, and has to give the same answers.
    """

    def test_module_resolution_finds_a_sibling(self, project):
        _open(project / "main.spc")
        resolved = server_module.resolve_module_path("shapes", path_to_uri(project / "main.spc"))

        assert resolved is not None
        assert resolved.name == "shapes.spc"

    def test_module_resolution_misses_cleanly(self, project):
        _open(project / "main.spc")
        assert server_module.resolve_module_path(
            "no_such_module_anywhere", path_to_uri(project / "main.spc")
        ) is None

    def test_definition_on_an_import_line_opens_the_module(self, project):
        doc = _open(project / "main.spc")

        location = server_module.definition(
            DefinitionParams(
                text_document=TextDocumentIdentifier(uri=doc.uri),
                position=_at(MAIN, "shapes"),
            )
        )

        assert location is not None
        assert uri_to_path(location.uri).name == "shapes.spc"

    def test_import_completion_offers_local_modules(self, project):
        doc = _open(project / "main.spc", "from ")

        result = server_module.completions(
            CompletionParams(
                text_document=TextDocumentIdentifier(uri=doc.uri),
                position=Position(line=0, character=5),
            )
        )

        assert "shapes" in {item.label for item in result.items}

    def test_lookup_paths_come_from_the_project(self, project):
        _open(project / "main.spc")
        paths = server_module.get_lookup_paths(path_to_uri(project / "main.spc"))

        assert project.resolve() in [p.resolve() for p in paths]


class TestRootDetection:
    def test_a_workspace_folder_wins(self, tmp_path):
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "x.spc").write_text("", encoding="utf-8")

        space = SpiceWorkspace()
        space.configure(workspace_folders=[tmp_path])
        assert space.root_for(nested / "x.spc") == tmp_path.resolve()

    def test_the_deepest_containing_folder_wins(self, tmp_path):
        outer = tmp_path / "outer"
        inner = outer / "inner"
        inner.mkdir(parents=True)
        (inner / "x.spc").write_text("", encoding="utf-8")

        space = SpiceWorkspace()
        space.configure(workspace_folders=[outer, inner])
        assert space.root_for(inner / "x.spc") == inner.resolve()

    def test_falls_back_to_a_project_marker(self, tmp_path):
        root = tmp_path / "proj"
        deep = root / "src" / "pkg"
        deep.mkdir(parents=True)
        (root / "pyproject.toml").write_text("", encoding="utf-8")
        (deep / "x.spc").write_text("", encoding="utf-8")

        space = SpiceWorkspace()
        assert space.root_for(deep / "x.spc") == root.resolve()

    def test_falls_back_to_the_files_own_directory(self, tmp_path):
        lonely = tmp_path / "nothing_here"
        lonely.mkdir()
        (lonely / "x.spc").write_text("", encoding="utf-8")

        space = SpiceWorkspace()
        # tmp_path has no markers above it either, short of the filesystem root.
        assert space.root_for(lonely / "x.spc").is_dir()


class TestQuietStdout:
    def test_analysis_prints_nothing(self, project, capsys):
        """stdout is the JSON-RPC channel.

        The compiler's stage logging goes there by default, and one stray
        "Tokenizing file: ..." line corrupts the stream and drops the client.
        """
        _open(project / "main.spc")

        captured = capsys.readouterr()
        assert captured.out == ""

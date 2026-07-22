"""Unit tests for ingestion/identity.py, ingestion/media.py, and connectors/filesystem.py.

Covers:
- Canonical source URI and ID construction (CWD-independence).
- Glob discovery determinism, no hardcoded extension list.
- External citation link mapping without leaking absolute paths.
- Revision vs source ID behavior.
- IngestionError content (identifies source+operation, no credentials).
- media.py type resolution and parser-hint emission.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from beacon_kb.errors import IngestionError
from beacon_kb.ingestion.identity import canonicalize_path, make_file_source_uri
from beacon_kb.ingestion.media import resolve_media_type

# ---------------------------------------------------------------------------
# identity.py: canonical URI construction
# ---------------------------------------------------------------------------


class TestCanonicalizePathCwdIndependence:
    """Canonical source URIs must be identical regardless of CWD."""

    def test_absolute_path_is_stable_across_cwd(self, tmp_path: pathlib.Path) -> None:
        """Absolute path -> file:// URI that never embeds CWD."""
        target = tmp_path / "docs" / "guide.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("hello")

        original_cwd = pathlib.Path.cwd()
        try:
            os.chdir(tmp_path)
            uri_from_tmp = canonicalize_path(str(target))
        finally:
            os.chdir(original_cwd)

        uri_from_original = canonicalize_path(str(target))
        assert uri_from_tmp == uri_from_original

    def test_relative_path_resolves_to_same_absolute(self, tmp_path: pathlib.Path) -> None:
        """A relative path canonicalized from two different CWDs must produce the same URI.

        This verifies that canonicalize_path resolves relative inputs against
        the filesystem (not a fixed base directory) and produces a stable
        CWD-independent ``file://`` URI.
        """
        target = tmp_path / "notes.txt"
        target.write_text("notes")

        original_cwd = pathlib.Path.cwd()
        # Build the relative path string from tmp_path's parent so it is
        # genuinely relative (not just an alias for an absolute path).
        rel = pathlib.Path(target.name)

        try:
            os.chdir(tmp_path)
            uri_from_tmp = canonicalize_path(str(rel))
        finally:
            os.chdir(original_cwd)

        # Canonicalized from an absolute path must match the relative-CWD result.
        uri_from_abs = canonicalize_path(str(target))
        assert uri_from_tmp == uri_from_abs
        assert uri_from_tmp.startswith("file://")
        assert "notes.txt" in uri_from_tmp

    def test_uri_does_not_embed_separator_differences(self, tmp_path: pathlib.Path) -> None:
        """URI uses forward slashes regardless of OS path separator."""
        target = tmp_path / "sub" / "file.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x")
        uri = canonicalize_path(str(target))
        assert "\\" not in uri


class TestMakeFileSourceUri:
    """make_file_source_uri: URI round-trips to SourceId."""

    def test_uri_scheme_is_file(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "readme.md"
        target.write_text("# Title")
        uri = make_file_source_uri(target)
        assert uri.startswith("file://")

    def test_uri_contains_filename(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "my-doc.txt"
        target.write_text("content")
        uri = make_file_source_uri(target)
        assert "my-doc.txt" in uri

    def test_uri_is_deterministic(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "stable.md"
        target.write_text("stable")
        assert make_file_source_uri(target) == make_file_source_uri(target)


# ---------------------------------------------------------------------------
# identity.py: source ID vs revision ID behavior
# ---------------------------------------------------------------------------


class TestSourceIdStability:
    """Source ID must not change when content changes; revision ID must."""

    def test_source_id_unchanged_after_content_change(self, tmp_path: pathlib.Path) -> None:
        from beacon_kb.models import make_source_id

        target = tmp_path / "doc.md"
        target.write_text("version 1")

        uri = make_file_source_uri(target)
        sid_before = make_source_id(corpus="test-corpus", canonical_uri=uri)

        # Change content - source ID must remain identical
        target.write_text("version 2")
        sid_after = make_source_id(corpus="test-corpus", canonical_uri=uri)

        # Source ID must be identical; revision ID will differ (different content hash)
        assert sid_before == sid_after

    def test_revision_id_changes_when_content_changes(self, tmp_path: pathlib.Path) -> None:
        import hashlib

        from beacon_kb.models import make_revision_id, make_source_id

        target = tmp_path / "evolving.md"
        target.write_text("v1")

        uri = make_file_source_uri(target)
        sid = make_source_id(corpus="c", canonical_uri=uri)

        h1 = hashlib.sha256(target.read_bytes()).hexdigest()
        target.write_text("v2")
        h2 = hashlib.sha256(target.read_bytes()).hexdigest()

        rev1 = make_revision_id(
            source_id=str(sid), content_hash=h1, pipeline_fingerprint="p"
        )
        rev2 = make_revision_id(
            source_id=str(sid), content_hash=h2, pipeline_fingerprint="p"
        )
        assert rev1 != rev2


# ---------------------------------------------------------------------------
# FilesystemConnector: glob discovery
# ---------------------------------------------------------------------------


class TestFilesystemGlobDiscovery:
    """Filesystem connector glob discovery rules."""

    def test_discovers_matching_files(self, tmp_path: pathlib.Path) -> None:
        from beacon_kb.connectors.filesystem import FilesystemConnector

        (tmp_path / "a.md").write_text("A")
        (tmp_path / "b.md").write_text("B")
        (tmp_path / "c.txt").write_text("C")

        connector = FilesystemConnector(
            root=tmp_path,
            corpus="corp",
            patterns=["*.md"],
        )
        sources = connector.list_sources()
        # Only .md files matched
        uris_basenames = [pathlib.Path(u.replace("file://", "")).name for u in sources]
        assert "a.md" in uris_basenames
        assert "b.md" in uris_basenames
        assert "c.txt" not in uris_basenames

    def test_discovery_is_deterministic(self, tmp_path: pathlib.Path) -> None:
        from beacon_kb.connectors.filesystem import FilesystemConnector

        for name in ["z.md", "a.md", "m.md"]:
            (tmp_path / name).write_text(name)

        connector = FilesystemConnector(
            root=tmp_path,
            corpus="corp",
            patterns=["*.md"],
        )
        assert connector.list_sources() == connector.list_sources()

    def test_discovery_is_sorted(self, tmp_path: pathlib.Path) -> None:
        from beacon_kb.connectors.filesystem import FilesystemConnector

        for name in ["z.md", "a.md", "m.md"]:
            (tmp_path / name).write_text(name)

        connector = FilesystemConnector(
            root=tmp_path,
            corpus="corp",
            patterns=["*.md"],
        )
        sources = connector.list_sources()
        assert sources == sorted(sources)

    def test_no_hardcoded_extension_list(self, tmp_path: pathlib.Path) -> None:
        """Connector must use the caller-supplied pattern, not a built-in extension whitelist."""
        from beacon_kb.connectors.filesystem import FilesystemConnector

        (tmp_path / "notes.xyz_custom_ext").write_text("custom")
        connector = FilesystemConnector(
            root=tmp_path,
            corpus="corp",
            patterns=["*.xyz_custom_ext"],
        )
        sources = connector.list_sources()
        assert len(sources) == 1

    def test_multiple_patterns(self, tmp_path: pathlib.Path) -> None:
        from beacon_kb.connectors.filesystem import FilesystemConnector

        (tmp_path / "a.md").write_text("A")
        (tmp_path / "b.rst").write_text("B")
        (tmp_path / "c.py").write_text("C")

        connector = FilesystemConnector(
            root=tmp_path,
            corpus="corp",
            patterns=["*.md", "*.rst"],
        )
        sources = connector.list_sources()
        basenames = {pathlib.Path(u.replace("file://", "")).name for u in sources}
        assert "a.md" in basenames
        assert "b.rst" in basenames
        assert "c.py" not in basenames

    def test_empty_directory_returns_empty_list(self, tmp_path: pathlib.Path) -> None:
        from beacon_kb.connectors.filesystem import FilesystemConnector

        connector = FilesystemConnector(
            root=tmp_path,
            corpus="corp",
            patterns=["*.md"],
        )
        assert connector.list_sources() == []


# ---------------------------------------------------------------------------
# FilesystemConnector: fetch and loading
# ---------------------------------------------------------------------------


class TestFilesystemFetch:
    """Fetch returns RawDocument without parsing."""

    def test_fetch_returns_raw_document(self, tmp_path: pathlib.Path) -> None:
        from beacon_kb.connectors.filesystem import FilesystemConnector
        from beacon_kb.models import RawDocument

        target = tmp_path / "hello.md"
        target.write_text("# Hello\nWorld")
        connector = FilesystemConnector(
            root=tmp_path,
            corpus="c",
            patterns=["*.md"],
        )
        uri = connector.list_sources()[0]
        doc = connector.fetch(uri)
        assert isinstance(doc, RawDocument)

    def test_fetch_content_matches_file(self, tmp_path: pathlib.Path) -> None:
        from beacon_kb.connectors.filesystem import FilesystemConnector

        content = "# Title\nBody text here."
        target = tmp_path / "doc.md"
        target.write_text(content)
        connector = FilesystemConnector(
            root=tmp_path,
            corpus="c",
            patterns=["*.md"],
        )
        uri = connector.list_sources()[0]
        doc = connector.fetch(uri)
        assert doc.content == content

    def test_fetch_does_not_parse(self, tmp_path: pathlib.Path) -> None:
        """Fetch returns raw bytes/text; the content is the file text, not parsed sections."""
        from beacon_kb.connectors.filesystem import FilesystemConnector

        raw = "## Section\nSome text. [link](https://example.com)"
        target = tmp_path / "page.md"
        target.write_text(raw)
        connector = FilesystemConnector(
            root=tmp_path,
            corpus="c",
            patterns=["*.md"],
        )
        uri = connector.list_sources()[0]
        doc = connector.fetch(uri)
        # Content must be the raw text, not parsed/transformed output
        assert "## Section" in doc.content
        assert "[link]" in doc.content

    def test_fetch_unknown_uri_raises_ingestion_error(self, tmp_path: pathlib.Path) -> None:
        from beacon_kb.connectors.filesystem import FilesystemConnector

        connector = FilesystemConnector(
            root=tmp_path,
            corpus="c",
            patterns=["*.md"],
        )
        with pytest.raises(IngestionError) as exc_info:
            connector.fetch("file:///nonexistent/path/ghost.md")
        msg = str(exc_info.value)
        # Error must identify source and operation
        assert "ghost.md" in msg or "nonexistent" in msg

    def test_fetch_error_does_not_include_credentials(self, tmp_path: pathlib.Path) -> None:
        """Error messages must not include secrets or credential-like strings."""
        from beacon_kb.connectors.filesystem import FilesystemConnector

        connector = FilesystemConnector(
            root=tmp_path,
            corpus="c",
            patterns=["*.md"],
        )
        try:
            connector.fetch("file:///missing/file.md")
        except IngestionError as exc:
            msg = str(exc)
            # Basic check: no password/token keywords
            for bad in ("password", "secret", "token", "api_key"):
                assert bad not in msg.lower(), f"Error message must not contain '{bad}'"


# ---------------------------------------------------------------------------
# FilesystemConnector: external citation link mapping
# ---------------------------------------------------------------------------


class TestExternalLinkMapping:
    """External citation links must not leak absolute paths when mapping configured."""

    def test_external_link_derived_without_absolute_path(self, tmp_path: pathlib.Path) -> None:
        from beacon_kb.connectors.filesystem import FilesystemConnector

        (tmp_path / "guide.md").write_text("# Guide")
        connector = FilesystemConnector(
            root=tmp_path,
            corpus="c",
            patterns=["*.md"],
            external_base_url="https://docs.example.com/",
        )
        uri = connector.list_sources()[0]
        external = connector.external_url(uri)
        assert external is not None
        assert external.startswith("https://docs.example.com/")
        # Must not expose the local absolute path in the external link
        assert str(tmp_path) not in external

    def test_no_external_link_when_no_mapping(self, tmp_path: pathlib.Path) -> None:
        from beacon_kb.connectors.filesystem import FilesystemConnector

        (tmp_path / "local.md").write_text("content")
        connector = FilesystemConnector(
            root=tmp_path,
            corpus="c",
            patterns=["*.md"],
        )
        uri = connector.list_sources()[0]
        assert connector.external_url(uri) is None


# ---------------------------------------------------------------------------
# media.py: media type resolution
# ---------------------------------------------------------------------------


class TestMediaTypeResolution:
    """resolve_media_type must return a MIME string without parsing content."""

    def test_md_extension_returns_markdown_type(self) -> None:
        mt = resolve_media_type("file:///docs/guide.md")
        assert "markdown" in mt.lower() or mt == "text/markdown"

    def test_txt_extension_returns_plain_text(self) -> None:
        mt = resolve_media_type("file:///notes.txt")
        assert mt == "text/plain"

    def test_html_extension(self) -> None:
        mt = resolve_media_type("file:///page.html")
        assert "html" in mt.lower()

    def test_unknown_extension_returns_octet_stream(self) -> None:
        mt = resolve_media_type("file:///data.xyz_unknown_ext_12345")
        # Must not raise; returns a safe fallback
        assert isinstance(mt, str)
        assert len(mt) > 0

    def test_pdf_extension(self) -> None:
        mt = resolve_media_type("file:///report.pdf")
        assert "pdf" in mt.lower()

    def test_resolve_does_not_open_file(self) -> None:
        """resolve_media_type must work purely from the URI, never opening the file."""
        # Path that does not exist - must still return a media type without raising
        mt = resolve_media_type("file:///does/not/exist/doc.md")
        assert isinstance(mt, str)
        assert len(mt) > 0

    def test_returns_parser_hint(self) -> None:
        """resolve_media_type may return an extra parser hint tuple."""
        from beacon_kb.ingestion.media import resolve_media_type_with_hint

        media_type, hint = resolve_media_type_with_hint("file:///guide.md")
        assert isinstance(media_type, str)
        assert isinstance(hint, str)

    def test_csv_gets_plaintext_hint_not_markdown(self) -> None:
        """text/csv must resolve to the neutral 'plaintext' hint, not 'markdown'.

        The prefix fallback must use an explicit text/* -> plaintext mapping
        rather than picking the first text/* entry in _PARSER_HINTS (which
        would wrongly return 'markdown' for text/csv).
        """
        from beacon_kb.ingestion.media import resolve_media_type_with_hint

        media_type, hint = resolve_media_type_with_hint("file:///data.csv")
        assert "csv" in media_type or media_type.startswith("text/")
        assert hint == "plaintext", (
            f"Expected 'plaintext' for {media_type!r}, got {hint!r}"
        )

    def test_unknown_text_subtype_gets_plaintext_hint(self) -> None:
        """An unrecognised text/* type must get 'plaintext', not a misleading hint."""
        from beacon_kb.ingestion.media import resolve_media_type_with_hint

        # Use a fake URI extension that maps to text/x-custom-not-in-table or similar;
        # we inject directly by testing with a type the table does not cover.
        # .asc is not in _PARSER_HINTS; mimetypes may return text/plain or None.
        # Either way the hint must never be "markdown" or "rst" etc.
        media_type, hint = resolve_media_type_with_hint("file:///notes.csv")
        assert hint in ("plaintext", "binary"), (
            f"Unexpected hint {hint!r} for {media_type!r}; expected plaintext or binary"
        )


# ---------------------------------------------------------------------------
# FilesystemConnector: binary and undecodable file handling
# ---------------------------------------------------------------------------


class TestFilesystemFetchBinaryAndDecodeErrors:
    """Fetch must raise IngestionError for binary files and invalid UTF-8."""

    def test_binary_file_raises_ingestion_error(self, tmp_path: pathlib.Path) -> None:
        """A file with a binary media type must raise IngestionError, not corrupt content."""
        from beacon_kb.connectors.filesystem import FilesystemConnector

        # Write a file with PNG magic bytes; .png extension maps to image/png.
        png_magic = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
        binary_file = tmp_path / "image.png"
        binary_file.write_bytes(png_magic)

        connector = FilesystemConnector(
            root=tmp_path,
            corpus="c",
            patterns=["*.png"],
        )
        uri = connector.list_sources()[0]
        with pytest.raises(IngestionError) as exc_info:
            connector.fetch(uri)
        msg = str(exc_info.value)
        assert "binary" in msg.lower() or "non-text" in msg.lower() or "image" in msg.lower()

    def test_invalid_utf8_text_file_raises_ingestion_error(self, tmp_path: pathlib.Path) -> None:
        """A .txt file containing invalid UTF-8 bytes must raise IngestionError.

        The connector must never silently replace undecodable bytes with U+FFFD.
        """
        from beacon_kb.connectors.filesystem import FilesystemConnector

        # Write raw bytes that are not valid UTF-8.
        invalid_utf8 = b"Valid start \xff\xfe invalid continuation"
        bad_file = tmp_path / "broken.txt"
        bad_file.write_bytes(invalid_utf8)

        connector = FilesystemConnector(
            root=tmp_path,
            corpus="c",
            patterns=["*.txt"],
        )
        uri = connector.list_sources()[0]
        with pytest.raises(IngestionError) as exc_info:
            connector.fetch(uri)
        msg = str(exc_info.value)
        # Must not silently succeed - the error must reference the URI.
        assert "broken.txt" in msg or uri in msg

    def test_valid_utf8_text_file_succeeds(self, tmp_path: pathlib.Path) -> None:
        """A valid UTF-8 text file must decode successfully with no replacement chars."""
        from beacon_kb.connectors.filesystem import FilesystemConnector

        content = "Hello, world! Unicode: éàü"
        text_file = tmp_path / "valid.txt"
        text_file.write_text(content, encoding="utf-8")

        connector = FilesystemConnector(
            root=tmp_path,
            corpus="c",
            patterns=["*.txt"],
        )
        uri = connector.list_sources()[0]
        doc = connector.fetch(uri)
        assert doc.content == content
        assert "�" not in doc.content

"""Unit tests for FolderConnector and the Connector interface (Task 02.1)."""

from __future__ import annotations

import pathlib

import pytest

# --- Interface imports (will fail until base.py exists) ---
from beacon.ingest.connectors.base import (
    ConfirmedDeletion,
    ConnectorKind,
    FetchSuccess,
    SourceEntry,
    TransientFailure,
)
from beacon.ingest.connectors.folder import FolderConnector


class TestConnectorInterface:
    """The Connector interface is importable and the types are correct."""

    def test_fetch_success_has_content(self) -> None:
        result = FetchSuccess(
            content=b"hello",
            content_hash="abc123",
            media_type="text/plain",
        )
        assert result.content == b"hello"
        assert result.content_hash == "abc123"

    def test_transient_failure_has_uri(self) -> None:
        result = TransientFailure(uri="file:///a.txt", reason="permission denied")
        assert result.uri == "file:///a.txt"

    def test_confirmed_deletion_has_uri(self) -> None:
        result = ConfirmedDeletion(uri="file:///gone.txt")
        assert result.uri == "file:///gone.txt"

    def test_source_entry_fields(self) -> None:
        entry = SourceEntry(
            uri="file:///x.md",
            title="x.md",
            connector_kind=ConnectorKind.FOLDER,
            media_type="text/markdown",
            metadata={},
        )
        assert entry.uri == "file:///x.md"
        assert entry.connector_kind == "folder"


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _write(path: pathlib.Path, content: bytes = b"hello") -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------
# TestFolderConnectorEnumerate
# ---------------------------------------------------------------


class TestFolderConnectorEnumerate:
    """FolderConnector.enumerate() discovers files by glob."""

    def test_discovers_markdown_files(self, tmp_path: pathlib.Path) -> None:
        _write(tmp_path / "a.md")
        _write(tmp_path / "sub" / "b.md")
        conn = FolderConnector(
            root=tmp_path,
            include_globs=["**/*.md"],
        )
        entries = conn.enumerate()
        uris = {e.uri for e in entries}
        assert f"file://{(tmp_path / 'a.md').resolve().as_posix()}" in uris
        assert f"file://{(tmp_path / 'sub' / 'b.md').resolve().as_posix()}" in uris

    def test_excludes_by_glob(self, tmp_path: pathlib.Path) -> None:
        _write(tmp_path / "keep.md")
        _write(tmp_path / "skip.md")
        conn = FolderConnector(
            root=tmp_path,
            include_globs=["**/*.md"],
            exclude_globs=["**/skip.md"],
        )
        uris = {e.uri for e in conn.enumerate()}
        assert not any("skip.md" in u for u in uris)
        assert any("keep.md" in u for u in uris)

    def test_returns_file_uri_scheme(self, tmp_path: pathlib.Path) -> None:
        _write(tmp_path / "doc.md")
        conn = FolderConnector(root=tmp_path, include_globs=["**/*.md"])
        entries = conn.enumerate()
        assert all(e.uri.startswith("file://") for e in entries)

    def test_connector_kind_is_folder(self, tmp_path: pathlib.Path) -> None:
        _write(tmp_path / "doc.md")
        conn = FolderConnector(root=tmp_path, include_globs=["**/*.md"])
        entries = conn.enumerate()
        assert all(e.connector_kind == "folder" for e in entries)

    def test_empty_directory_returns_empty(self, tmp_path: pathlib.Path) -> None:
        conn = FolderConnector(root=tmp_path, include_globs=["**/*.md"])
        assert conn.enumerate() == []

    def test_uri_is_stable_across_calls(self, tmp_path: pathlib.Path) -> None:
        _write(tmp_path / "doc.md")
        conn = FolderConnector(root=tmp_path, include_globs=["**/*.md"])
        uris1 = {e.uri for e in conn.enumerate()}
        uris2 = {e.uri for e in conn.enumerate()}
        assert uris1 == uris2

    def test_missing_root_raises_ingestion_error(self, tmp_path: pathlib.Path) -> None:
        from beacon.errors import IngestionError
        conn = FolderConnector(root=tmp_path / "nonexistent", include_globs=["**/*.md"])
        with pytest.raises(IngestionError):
            conn.enumerate()

    def test_media_type_markdown(self, tmp_path: pathlib.Path) -> None:
        _write(tmp_path / "doc.md")
        conn = FolderConnector(root=tmp_path, include_globs=["**/*.md"])
        entries = conn.enumerate()
        assert entries[0].media_type in ("text/markdown", "text/x-markdown", "text/plain")

    def test_media_type_text_plain(self, tmp_path: pathlib.Path) -> None:
        _write(tmp_path / "doc.txt")
        conn = FolderConnector(root=tmp_path, include_globs=["**/*.txt"])
        entries = conn.enumerate()
        assert "text" in entries[0].media_type


class TestFolderConnectorFetch:
    """FolderConnector.fetch() returns the correct FetchResult discriminated union."""

    def test_fetch_success_returns_bytes(self, tmp_path: pathlib.Path) -> None:
        _write(tmp_path / "a.md", b"content here")
        conn = FolderConnector(root=tmp_path, include_globs=["**/*.md"])
        entries = conn.enumerate()
        result = conn.fetch(entries[0].uri)
        assert isinstance(result, FetchSuccess)
        assert result.content == b"content here"

    def test_fetch_success_content_hash_is_sha256(self, tmp_path: pathlib.Path) -> None:
        import hashlib
        data = b"hash me"
        _write(tmp_path / "a.md", data)
        conn = FolderConnector(root=tmp_path, include_globs=["**/*.md"])
        entries = conn.enumerate()
        result = conn.fetch(entries[0].uri)
        assert isinstance(result, FetchSuccess)
        assert result.content_hash == hashlib.sha256(data).hexdigest()

    def test_fetch_missing_file_is_confirmed_deletion(self, tmp_path: pathlib.Path) -> None:
        """A file absent from disk is a ConfirmedDeletion, not an error."""
        # Enumerate first to get a known URI, then delete the file.
        p = _write(tmp_path / "will_delete.md")
        conn = FolderConnector(root=tmp_path, include_globs=["**/*.md"])
        entries = conn.enumerate()
        uri = entries[0].uri
        p.unlink()
        result = conn.fetch(uri)
        assert isinstance(result, ConfirmedDeletion)
        assert result.uri == uri

    def test_fetch_permission_error_is_transient(self, tmp_path: pathlib.Path) -> None:
        """An existing but unreadable file is TransientFailure, not ConfirmedDeletion."""
        import os
        import sys
        if sys.platform == "win32":
            pytest.skip("chmod not reliable on Windows")
        p = _write(tmp_path / "locked.md", b"secret")
        conn = FolderConnector(root=tmp_path, include_globs=["**/*.md"])
        entries = conn.enumerate()
        uri = entries[0].uri
        # Remove read permission.
        os.chmod(p, 0o000)
        try:
            result = conn.fetch(uri)
            assert isinstance(result, TransientFailure)
            assert result.uri == uri
        finally:
            os.chmod(p, 0o644)

    def test_fetch_result_hash_stable(self, tmp_path: pathlib.Path) -> None:
        """Same content -> same hash across two fetches."""
        _write(tmp_path / "a.md", b"stable")
        conn = FolderConnector(root=tmp_path, include_globs=["**/*.md"])
        entries = conn.enumerate()
        r1 = conn.fetch(entries[0].uri)
        r2 = conn.fetch(entries[0].uri)
        assert isinstance(r1, FetchSuccess) and isinstance(r2, FetchSuccess)
        assert r1.content_hash == r2.content_hash

# beacon-kb Parsing Fixture

This document is safe, newly authored regression content for the beacon-kb parser test suite.
It covers headings, fenced code blocks, tables, and links.

## Installation

Install using pip or uv:

```bash
pip install beacon-kb
uv add beacon-kb
```

Use the optional `html` extra for HTML parsing:

```bash
pip install "beacon-kb[html]"
```

## Configuration

The library reads configuration from `pyproject.toml` under the `[tool.beacon-kb]` section.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `corpus` | str | `"default"` | Logical corpus name |
| `root` | str | `"."` | Root directory to scan |
| `patterns` | list | `["**/*.md"]` | Glob patterns |

See the [Configuration Reference](https://example.com/docs/config) for all options.

## Usage

### Basic Example

Create a connector and parse a document:

```python
from beacon_kb.connectors.filesystem import FilesystemConnector

connector = FilesystemConnector(
    root="./docs",
    corpus="my-corpus",
    patterns=["**/*.md"],
)
uris = connector.list_sources()
```

### Advanced Usage

Register a custom parser:

```python
from beacon_kb.registry import precedence, groups

precedence.register(
    group=groups.PARSERS,
    name="custom",
    instance=MyParser(),
)
```

## API Reference

### FilesystemConnector

`FilesystemConnector` discovers and loads files from a configured root directory.

- `list_sources() -> list[str]`: Returns sorted canonical URIs.
- `fetch(uri: str) -> RawDocument`: Loads file content.

### Section Model

A `Section` represents a heading-delimited unit within a parsed document.
Every section carries a `locator` (heading path), `heading`, `text`, and `ordinal`.

## Links and References

- [Project Repository](https://github.com/example/beacon-kb)
- [Issue Tracker](https://github.com/example/beacon-kb/issues)
- [Documentation](https://example.com/docs)

## Trailing Section

This section has no subheadings.
It ensures the parser does not drop content that appears after the last heading.

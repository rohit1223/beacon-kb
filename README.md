# beacon-kb

A modular, plugin-driven Python library for building agentic retrieval-augmented generation (RAG) pipelines.
beacon-kb ships a composable stage-by-stage pipeline - connectors, parsers, chunkers, embedders, stores, retrievers, re-rankers, generators, and graders - with every stage replaceable through a lightweight entry-point registry.

## Installation modes

### Base install (offline, no model downloads)

```bash
pip install beacon-kb
```

The base install pulls only NumPy and uses SQLite FTS5 (bundled with CPython) for keyword retrieval.
No credentials, no network calls, and no model downloads are required at import time.

### Optional extras

| Extra | What it adds |
|-------|-------------|
| `html` | HTML parsing via BeautifulSoup4 and lxml |
| `pdf` | PDF extraction via pypdf |
| `web` | HTTP connector via httpx |
| `confluence` | Atlassian Confluence connector |
| `remote` | Generic remote/API connector (httpx) |
| `local` | Local filesystem connector (stdlib only) |
| `mcp` | Model Context Protocol integration |
| `dev` | Ruff, mypy, pytest, pytest-cov, build |

Install one or more extras together:

```bash
pip install "beacon-kb[html,pdf,web]"
```

## Five-minute offline quickstart

```python
import beacon_kb

print(beacon_kb.__version__)
```

No network access is needed.
The package is fully typed (PEP 561 `py.typed` marker is included).

## Console scripts

Two equivalent entry points are provided:

```bash
beacon-kb --help
bkb --help
```

Both resolve to `beacon_kb.cli:main`.

## Plugin system

Third-party plugins register themselves via Python entry-point groups:

- `beacon_kb.connectors`
- `beacon_kb.parsers`
- `beacon_kb.chunkers`
- `beacon_kb.embedders`
- `beacon_kb.stores`
- `beacon_kb.retrievers`
- `beacon_kb.fusion`
- `beacon_kb.rerankers`
- `beacon_kb.generators`
- `beacon_kb.token_counters`
- `beacon_kb.planners`
- `beacon_kb.graders`
- `beacon_kb.routers`

## License

MIT - see [LICENSE](LICENSE).

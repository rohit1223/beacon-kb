"""Integration test configuration for the beacon package.

Integration tests require real external services or file I/O that is not
available in a pure unit-test environment.  Mark each test with the
``integration`` pytest marker so they can be excluded in CI runs that lack
those services::

    @pytest.mark.integration
    def test_qdrant_round_trip() -> None:
        ...

To run integration tests only::

    pytest -m integration tests/beacon/integration/

To skip them (default CI behaviour)::

    pytest -m "not integration" tests/beacon/

The ``integration`` marker is registered in ``pyproject.toml`` under
``[tool.pytest.ini_options] markers``.
"""

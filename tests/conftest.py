"""Test tiering: tests are unit by default; lake/network ones opt into integration."""

import pytest


def pytest_collection_modifyitems(config, items):
    """Any test not explicitly marked integration counts as a unit test."""
    for item in items:
        if "integration" not in item.keywords:
            item.add_marker(pytest.mark.unit)

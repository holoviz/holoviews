import importlib
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parent
ALL_JSON = HERE / "all.json"

with ALL_JSON.open("r", encoding="utf-8") as f:
    EXPECTED = json.load(f)


@pytest.mark.parametrize("module_name", sorted(EXPECTED))
def test_all_matches_snapshot(module_name):
    module = importlib.import_module(module_name)
    expected = EXPECTED[module_name]

    if expected is None:
        assert not hasattr(module, "__all__")
        return

    assert hasattr(module, "__all__")
    assert set(module.__all__) == set(expected)

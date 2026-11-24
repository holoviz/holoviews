from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
PYPROJECT = ROOT / "pyproject.toml"
RECIPE = ROOT / "scripts" / "conda" / "recipe" / "recipe.yaml"

PYPI_TO_CONDA = {"matplotlib": "matplotlib-base"}

pytestmark = pytest.mark.skipif(
    not (PYPROJECT.exists() and RECIPE.exists()), reason="Requires a source checkout"
)


@pytest.fixture(scope="module")
def pyproject():
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


@pytest.fixture(scope="module")
def recipe():
    with RECIPE.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize(specs):
    reqs = []
    for spec in specs:
        if "${{" in spec:
            continue
        name, _, version = spec.partition(" ")
        reqs.append(f"{PYPI_TO_CONDA.get(name, name)} {version}".strip())
    return sorted(reqs)


def test_python_min(pyproject, recipe):
    assert f">={recipe['context']['python_min']}" == pyproject["project"]["requires-python"]


def test_host_requirements(pyproject, recipe):
    host = normalize(recipe["requirements"]["host"])
    for dep in normalize(pyproject["build-system"]["requires"]):
        assert dep in host


def test_run_requirements(pyproject, recipe):
    project = pyproject["project"]
    recommended = [
        dep
        for dep in project["optional-dependencies"]["recommended"]
        if dep.partition(" ")[0] in PYPI_TO_CONDA
    ]
    expected = normalize([*project["dependencies"], *recommended])
    assert normalize(recipe["requirements"]["run"]) == expected


def test_entry_points(pyproject, recipe):
    expected = sorted(f"{k} = {v}" for k, v in pyproject["project"]["scripts"].items())
    assert sorted(recipe["build"]["python"]["entry_points"]) == expected

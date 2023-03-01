#!/usr/bin/env python

import json
import os

from setuptools import setup, find_namespace_packages


def get_setup_version(reponame):
    """
    Helper to get the current version from either git describe or the
    .version file (if available).
    """
    basepath = os.path.split(__file__)[0]
    version_file_path = os.path.join(basepath, reponame, ".version")

    from param import version  # rely in pyproject.toml to install the right version
    return json.load(open(version_file_path))["version_string"]


def find_all_packages():
    regular_packages = find_namespace_packages(include=["holoviews*"])
    example_subpackages = [
        f"holoviews.examples.{p}"
        for p in find_namespace_packages(where="examples")
    ]
    return [*regular_packages, "holoviews.examples", *example_subpackages]


setup(
    version=get_setup_version("holoviews"),
    packages=find_all_packages(),
    package_dir={"holoviews.examples": "examples"},  # remap subpackage directory
    include_package_data=True,
)

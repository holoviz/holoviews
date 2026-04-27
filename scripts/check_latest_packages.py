from __future__ import annotations

import json
import os
import re
import sys
from importlib.metadata import version
from subprocess import DEVNULL, check_output

PYTHON_VERSION = sys.version_info[:2]
PLATFORM_SPECIFIERS = {"linux": "linux-64", "darwin": "osx-arm64", "win32": "win-64"}
PLATFORM = PLATFORM_SPECIFIERS[sys.platform]

if sys.stdout.isatty() or os.environ.get("GITHUB_ACTIONS"):
    GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
else:
    GREEN = RED = RESET = ""


def convert_int(x: str) -> tuple[int, ...]:
    return tuple(map(int, re.findall(r"\d+", x)))


def python_check(item):
    py_version = [
        py
        for py in item["depends"]
        if py.lower().startswith("python") and not py.lower().startswith("python-")
    ]

    if len(py_version) == 1:
        return PYTHON_VERSION >= convert_int(py_version[0])
    if len(py_version) == 2:
        # For compiled: python_abi 3.12.* *_cp312
        idx = int("abi" not in py_version[0])
        return PYTHON_VERSION == convert_int(py_version[idx].split("*")[0])
    return False


def main(*packages):
    all_latest = True
    for package in sorted(packages):
        out = check_output(
            ["pixi", "search", package, "--json", "--channel", "conda-forge"], stderr=DEVNULL
        )
        raw = json.loads(out)
        data = [*raw.get("noarch", ()), *raw.get(PLATFORM, ())]
        versions = {item["version"] for item in data if python_check(item)}
        latest_str = max(versions, key=convert_int)
        latest_int = convert_int(latest_str)
        current_str = version(package)
        current_int = convert_int(current_str)
        is_latest = current_int >= latest_int
        all_latest &= is_latest

        text_color = GREEN if is_latest else RED
        print(
            f"{text_color}Package: {package:<16} Current: {current_str:<16}\tLatest: {latest_str}{RESET}"
        )

    if not all_latest:
        sys.exit(1)


if __name__ == "__main__":
    main(*sys.argv[1:])

from __future__ import annotations

import sys
import typing as t
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def install_pre_commit_hook() -> None:
    """Install a pre-commit hook."""
    data = f"""#!/usr/bin/env bash
INSTALL_PYTHON={sys.executable}
ARGS=(hook-impl --config=.pre-commit-config.yaml --hook-type=pre-commit)
HERE="$(cd "$(dirname "$0")" && pwd)"
ARGS+=(--hook-dir "$HERE" -- "$@")
exec "$INSTALL_PYTHON" -m pre_commit "${{ARGS[@]}}"
"""
    if not Path(".git").exists():
        return

    path = Path(".git/hooks/pre-commit")
    if not path.exists():
        with path.open("w") as fid:
            fid.write(data)

    mode = path.stat().st_mode
    mode |= (mode & 0o444) >> 2  # copy R bits to X
    path.chmod(mode)


def install_pre_push_hook():
    data = """#!/usr/bin/env bash
set -euo pipefail

BRANCH=`git branch --show-current`

if [[ "$BRANCH" =~ ^(master|main)$ ]]; then
  echo
  echo "Prevented pushing to $BRANCH. Use --no-verify to bypass this pre-push hook."
  echo
  exit 1
fi

exit 0
"""
    if not Path(".git").exists():
        return

    path = Path(".git/hooks/pre-push")
    if not path.exists():
        with path.open("w") as fid:
            fid.write(data)

    mode = path.stat().st_mode
    mode |= (mode & 0o444) >> 2  # copy R bits to X
    path.chmod(mode)


class HoloviewsBuildHook(BuildHookInterface):
    """The hatch jupyter builder build hook."""

    PLUGIN_NAME = "install"

    def initialize(self, version: str, _: dict[str, t.Any]) -> None:
        """Initialize the plugin."""
        if self.target_name not in ["wheel", "sdist"]:
            return

        if version == "editable":
            install_pre_commit_hook()
            install_pre_push_hook()

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging", "pyyaml"]
# ///

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from functools import cache

import yaml
from packaging.version import InvalidVersion, Version  # noqa: TID251

COMMENT_MARKER = "<!-- pixi-lock-diff -->"

FILENAME_RE = re.compile(r"^(?P<name>.+)-(?P<version>[^-]+)-(?P<build>[^-]+)\.(?:conda|tar\.bz2)$")
VERSION_PART_RE = re.compile(r"\d+|\D+")


def git_show(revision: str, path: str) -> str:
    """Get a file's contents from a Git revision."""
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print(
            f"Could not read {path} from {revision}",
            file=sys.stderr,
        )
        sys.exit(1)

    return result.stdout


def git_rev_parse(revision: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", revision],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def load_environment_dependencies(content: str) -> dict:
    """Map each pixi environment name to the direct package names it pulls in."""
    manifest = tomllib.loads(content)

    feature_dependencies = {}
    for name, feature in manifest.get("feature", {}).items():
        names = feature.get("dependencies", ())
        feature_dependencies[name] = set(names)

    environments = {}
    for name, value in manifest.get("environments", {}).items():
        features = value if isinstance(value, list) else value.get("features", ())
        environments[name] = set().union(*(feature_dependencies.get(f, set()) for f in features))

    return environments


def load_direct_dependencies(base: str) -> dict:
    """Collect direct dependency names per pixi environment, from both base and HEAD pixi.toml."""
    with open("pixi.toml") as f:
        current = load_environment_dependencies(f.read())
    main = load_environment_dependencies(git_show(base, "pixi.toml"))

    return {
        env: current.get(env, set()) | main.get(env, set()) for env in set(current) | set(main)
    }


def parse_name_version(url: str) -> tuple[str, str, str] | None:
    match = FILENAME_RE.match(url.rsplit("/", 1)[-1])
    if not match:
        return None
    return match.group("name"), match.group("version"), match.group("build")


def load_environments(content: str) -> dict:
    """Extract package name/version per environment/platform from a pixi.lock."""
    lockfile = yaml.load(content, Loader=yaml.CSafeLoader)

    environments = {}

    for env_name, env in lockfile.get("environments", {}).items():
        platforms = {}

        for platform, refs in env.get("packages", {}).items():
            packages = {}

            for ref in refs:
                url = ref.get("conda") or ref.get("pypi")
                if not url:
                    continue

                parsed = parse_name_version(url)
                if not parsed:
                    continue

                name, version, build = parsed
                packages[name] = (version, build)

            platforms[platform] = packages

        environments[env_name] = platforms

    return environments


def version_key(version: str) -> list:
    """Split a version into ints/strs so parts compare numerically, e.g. '2' < '10'."""
    return [int(part) if part.isdigit() else part for part in VERSION_PART_RE.findall(version)]


def version_change(old_version: str | None, new_version: str | None) -> str:
    if old_version is None or new_version is None:
        return "Changed"
    try:
        # PEP440-aware so e.g. "2.4.2rc1" -> "2.4.2" is an upgrade, not a downgrade.
        return "Upgraded" if Version(new_version) > Version(old_version) else "Downgraded"
    except InvalidVersion:
        try:
            return (
                "Upgraded" if version_key(new_version) > version_key(old_version) else "Downgraded"
            )
        except TypeError:
            return "Changed"


def compare(main_platforms: dict, current_platforms: dict) -> list:
    changes = {}

    for platform in sorted(set(main_platforms) | set(current_platforms)):
        main_packages = main_platforms.get(platform, {})
        current_packages = current_platforms.get(platform, {})

        for package in sorted(set(main_packages) | set(current_packages)):
            old = main_packages.get(package)
            new = current_packages.get(package)

            old_version, old_build = old if old else (None, None)
            new_version, new_build = new if new else (None, None)

            if old is None:
                change = "Added"
            elif new is None:
                change = "Removed"
            elif old_version != new_version:
                change = version_change(old_version, new_version)
            elif old_build != new_build:
                change = "Build changed"
            else:
                continue

            if change == "Build changed":
                old_display = f"{old_version} ({old_build})"
                new_display = f"{new_version} ({new_build})"
            else:
                old_display = old_version or "-"
                new_display = new_version or "-"

            key = (package, old_display, new_display, change)
            changes.setdefault(key, []).append(platform)

    rows = [
        (package, ", ".join(platforms), old, new, change)
        for (package, old, new, change), platforms in changes.items()
    ]

    return sorted(rows)


def print_table(rows: list):
    print("| Package | Platform | Version (main) | Version (current) | Change |")
    print("|---|---|---|---|---|")

    for package, platform, old, new, change in sorted(rows):
        print(f"| {package} | {platform} | {old} | {new} | {change} |")


def print_markdown(env_name: str, rows: list, direct_dependencies: set, status: str | None = None):
    heading = f"### {env_name}" + (f" ({status})" if status else "")
    print(f"{heading}\n")

    direct_rows = [row for row in rows if row[0] in direct_dependencies]
    indirect_rows = [row for row in rows if row[0] not in direct_dependencies]

    if direct_rows:
        print_table(direct_rows)
        print()

    if indirect_rows:
        print("<details><summary>Indirect dependencies</summary>\n")
        print_table(indirect_rows)
        print("\n</details>")

    print()


def print_header(base: str):
    current_sha = git_rev_parse("HEAD")
    base_sha = git_rev_parse(base)

    print(COMMENT_MARKER)
    print("## Pixi.lock changes\n")
    print(f"Comparing this branch ({current_sha}) against base branch ({base_sha})\n")


@cache
def lockfile_unchanged(base: str) -> bool:
    """Cheap check to skip parsing when pixi.lock is identical to the base revision."""
    result = subprocess.run(
        ["git", "diff", "--quiet", base, "--", "pixi.lock"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def main():
    main = git_rev_parse("origin/main")
    base = os.environ.get("BASE_SHA") or main

    if lockfile_unchanged(base) or lockfile_unchanged(main):
        return

    print_header(base)
    main_content = git_show(base, "pixi.lock")

    with open("pixi.lock") as f:
        current_content = f.read()

    main_environments = load_environments(main_content)
    current_environments = load_environments(current_content)
    direct_dependencies = load_direct_dependencies(base)

    env_names = sorted(set(main_environments) | set(current_environments))

    any_changes = False

    for env_name in env_names:
        in_main = env_name in main_environments
        in_current = env_name in current_environments

        rows = compare(
            main_environments.get(env_name, {}),
            current_environments.get(env_name, {}),
        )

        if not rows:
            continue

        any_changes = True

        if not in_main:
            status = "added"
        elif not in_current:
            status = "removed"
        else:
            status = None

        print_markdown(env_name, rows, direct_dependencies.get(env_name, set()), status)

    if not any_changes:
        print("No changes to pixi.lock compared with main.")


if __name__ == "__main__":
    main()

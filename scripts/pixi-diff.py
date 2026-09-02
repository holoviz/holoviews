#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///

from __future__ import annotations

import os
import re
import subprocess
import sys

import tomllib
import yaml

LOCKFILE = "pixi.lock"
MANIFEST = "pixi.toml"
MAIN_BRANCH = "origin/main"

COMMENT_MARKER = "<!-- pixi-lock-diff -->"

FILENAME_RE = re.compile(r"^(?P<name>.+)-(?P<version>[^-]+)-(?P<build>[^-]+)\.(?:conda|tar\.bz2)$")
VERSION_PART_RE = re.compile(r"\d+|\D+")


def git_show(revision: str) -> str:
    """Get pixi.lock contents from a Git revision."""
    result = subprocess.run(
        ["git", "show", f"{revision}:{LOCKFILE}"],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print(
            f"Could not read {LOCKFILE} from {revision}",
            file=sys.stderr,
        )
        sys.exit(1)

    return result.stdout


def load_direct_dependencies() -> set:
    """Collect package names declared under any *dependencies table in pixi.toml."""
    with open(MANIFEST, "rb") as f:
        manifest = tomllib.load(f)

    names = set()

    def walk(table: dict):
        for key, value in table.items():
            if not isinstance(value, dict):
                continue
            if key.endswith("dependencies"):
                names.update(value)
            walk(value)

    walk(manifest)

    return names


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
        return "Upgraded" if version_key(new_version) > version_key(old_version) else "Downgraded"
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


def git_rev_parse(revision: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", revision],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def print_header(base: str):
    current_sha = git_rev_parse("HEAD")
    base_sha = git_rev_parse(base)

    print(COMMENT_MARKER)
    print("## Pixi.lock changes\n")
    print(f"Comparing `{MAIN_BRANCH}` ({base_sha}) against this branch ({current_sha}).\n")


def lockfile_unchanged(base: str) -> bool:
    """Cheap check to skip parsing when pixi.lock is identical to the base revision."""
    result = subprocess.run(
        ["git", "diff", "--quiet", base, "--", LOCKFILE],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def main():
    base = os.environ.get("BASE_SHA") or MAIN_BRANCH

    if lockfile_unchanged(base):
        return

    print_header(base)
    main_content = git_show(base)

    with open(LOCKFILE) as f:
        current_content = f.read()

    main_environments = load_environments(main_content)
    current_environments = load_environments(current_content)
    direct_dependencies = load_direct_dependencies()

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

        print_markdown(env_name, rows, direct_dependencies, status)

    if not any_changes:
        print("No changes to pixi.lock compared with main.")


if __name__ == "__main__":
    main()

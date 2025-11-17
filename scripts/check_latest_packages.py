import os
import sys
from datetime import date, datetime, timedelta
from importlib.metadata import version

import requests
from packaging.specifiers import SpecifierSet
from packaging.version import Version

PY_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

if sys.stdout.isatty() or os.environ.get("GITHUB_ACTIONS"):
    GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
else:
    GREEN = RED = RESET = ""


def compare_versions(version_str, constraint_str):
    try:
        # Convert version string to a Version object
        version = Version(version_str)

        # Convert constraint string to a SpecifierSet object
        constraint = SpecifierSet(constraint_str)

        # Check if the version satisfies the constraint
        if version in constraint:
            return True
        else:
            return False
    except Exception as e:
        return str(e)


def main(*packages):
    allowed_date = date.today() - timedelta(days=5)
    all_latest = True
    for package in sorted(packages):
        url = f"https://pypi.org/pypi/{package}/json"
        resp = requests.get(url, timeout=1).json()

        found = False
        for vrelease in sorted(resp["releases"], key=Version, reverse=True):
            if Version(vrelease).is_devrelease or Version(vrelease).is_prerelease:
                continue
            for info in resp["releases"][vrelease]:
                if not compare_versions(PY_VERSION, info['requires_python']):
                    continue

                latest = vrelease

                # Remove suffix because older Python versions does not support it
                latest_release_date = datetime.fromisoformat(
                    info["upload_time_iso_8601"].removesuffix("Z")
                ).date()
                found = True
                break
            if found:
                break
        else:
            raise RuntimeError('Could not find matching version')

        current = version(package)
        current_release_date = datetime.fromisoformat(
            resp["releases"][current][0]["upload_time_iso_8601"].removesuffix("Z")
        ).date()
        version_check = Version(current) >= Version(latest)
        date_check = latest_release_date >= allowed_date
        is_latest = version_check or date_check
        all_latest &= is_latest

        text_color = GREEN if is_latest else RED
        print(
            f"{text_color}Package: {package:<10} Current: {current:<7} ({current_release_date})\tLatest: {latest:<7} ({latest_release_date}){RESET}"
        )

    if not all_latest:
        sys.exit(1)


if __name__ == "__main__":
    main(*sys.argv[1:])

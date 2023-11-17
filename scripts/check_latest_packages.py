import sys
from datetime import date, datetime, timedelta

import requests
from packaging.version import Version


def main(*packages):
    allowed_date = date.today() - timedelta(days=5)
    GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
    all_latest = True
    for package in sorted(packages):
        url = f"https://pypi.org/pypi/{package}/json"
        resp = requests.get(url, timeout=20).json()
        latest = resp["info"]["version"]
        current = __import__(package).__version__

        latest_release_date = datetime.fromisoformat(
            resp["releases"][latest][0]["upload_time_iso_8601"]
        ).date()
        current_release_date = datetime.fromisoformat(
            resp["releases"][current][0]["upload_time_iso_8601"]
        ).date()

        version_check = Version(current) >= Version(latest)
        date_check = current_release_date >= allowed_date
        is_latest = version_check or date_check
        all_latest &= is_latest

        text_color = GREEN if is_latest else RED
        print(
            f"{text_color}Package: {package:<10} Current: {current:<7} ({current_release_date})\tLatest: {latest:<7} ({latest_release_date}){RESET}"
        )

    if not all_latest:
        sys.exit(1)


if __name__ == "__main__":
    main("numpy", "pandas")

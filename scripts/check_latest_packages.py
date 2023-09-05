import sys
import requests
from datetime import datetime, date, timedelta

from packaging.version import Version


def main(*packages):
    allowed_date = date.today() - timedelta(days=2)
    is_latest = True
    for package in packages:
        url = f"https://pypi.org/pypi/{package}/json"
        resp = requests.get(url).json()
        latest = resp["info"]["version"]
        current = __import__(package).__version__
        release_date = datetime.fromisoformat(
            resp["releases"][latest][0]["upload_time_iso_8601"]
        )
        version_check = Version(current) >= Version(latest)
        date_check = release_date.date() >= allowed_date
        is_latest &= version_check and date_check
        print(f"Package: {package:<10} Current: {current:<10} Latest: {latest:<10}")

    if not is_latest:
        sys.exit(1)


if __name__ == "__main__":
    main("numpy", "pandas")

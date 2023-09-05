import sys
import requests

from packaging.version import Version

def main(*packages):
    is_latest = True
    for package in packages:
        url = f"https://pypi.org/pypi/{package}/json"
        resp = requests.get(url).json()
        latest = Version(resp["info"]["version"])
        current = Version(__import__(package).__version__)
        is_latest &= current >= latest
        print(f"Package: {package:<10} Current: {current!s:<10} Latest: {latest!s:<10}")

    if not is_latest:
        sys.exit(1)

if __name__ == "__main__":
    main("numpy", "pandas")

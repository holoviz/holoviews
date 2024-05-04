"""
Script to sync tags from upstream repo to forked repo
"""

from subprocess import run

PACKAGE = "holoviews"

origin = run(["git", "remote", "get-url", "origin"], check=True, capture_output=True)
upstream = run(
    ["git", "remote", "get-url", "upstream"], check=False, capture_output=True
)

if upstream.returncode:
    is_http = b"http" in origin.stdout
    url = (
        f"https://github.com/holoviz/{PACKAGE}.git"
        if is_http
        else f"git@github.com:holoviz/{PACKAGE}.git"
    )
    print(f"Adding {url!r} as remote upstream")
    run(["git", "remote", "add", "upstream", url], check=True, capture_output=True)

print(f"Syncing tags from {PACKAGE} repo with forked repo")
run(["git", "fetch", "--tags", "upstream"], check=True, capture_output=True)
run(["git", "push", "--tags"], check=True, capture_output=True)

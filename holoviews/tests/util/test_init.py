from shutil import which
from subprocess import check_output
from textwrap import dedent


def test_no_blocklist_imports():
    check = """\
    import sys
    import holoviews as hv

    blocklist = {"panel", "IPython", "datashader", "ibis"}
    mods = blocklist & set(sys.modules)

    if mods:
        print(", ".join(mods), end="")
        """

    output = check_output([('python' if which('python') else 'python3'), '-c', dedent(check)])

    assert output == b""

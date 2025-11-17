import sys
from subprocess import check_output
from textwrap import dedent

import pytest


def test_no_blocklist_imports():
    check = """\
    import sys
    import holoviews as hv

    blocklist = {"panel", "IPython", "datashader", "ibis", "pandas"}
    mods = blocklist & set(sys.modules)

    if mods:
        print(", ".join(mods), end="")
    """

    output = check_output([sys.executable, '-c', dedent(check)])
    assert output == b""


def test_no_blocklist_imports_IPython():
    pytest.importorskip("IPython")

    check = """\
    import sys
    import holoviews as hv

    blocklist = {"panel", "datashader", "ibis", "pandas"}
    mods = blocklist & set(sys.modules)

    if mods:
        print(", ".join(mods), end="")
    """

    output = check_output([sys.executable, '-m', 'IPython', '-c', dedent(check)])
    assert output == b""

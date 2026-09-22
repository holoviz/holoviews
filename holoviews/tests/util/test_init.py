from __future__ import annotations

import sys
from subprocess import check_output
from textwrap import dedent

from .._deps import ipython_skip, mpl_skip


def test_no_blocklist_imports():
    check = """\
    import sys
    import holoviews as hv

    blocklist = {"panel", "IPython", "datashader", "ibis", "pandas"}
    mods = blocklist & set(sys.modules)

    if mods:
        print(", ".join(mods), end="")
    """

    output = check_output([sys.executable, "-c", dedent(check)])
    assert output == b""


@ipython_skip
def test_no_blocklist_imports_IPython():
    check = """\
    import sys
    import holoviews as hv

    blocklist = {"panel", "datashader", "ibis", "pandas"}
    mods = blocklist & set(sys.modules)

    if mods:
        print(", ".join(mods), end="")
    """

    output = check_output([sys.executable, "-m", "IPython", "-c", dedent(check)])
    assert output == b""


@mpl_skip
def test_mpl_cycle_colors_are_hex_strings():
    # Test for https://github.com/holoviz/holoviews/pull/6798
    check = """\
    import holoviews.plotting.bokeh
    import holoviews.plotting.mpl
    from holoviews.core.options import Cycle

    for name, colors in Cycle.default_cycles.items():
        for i, c in enumerate(colors):
            if not isinstance(c, str):
                print(f"{name}[{i}]={c!r}")
    """

    output = check_output([sys.executable, "-c", dedent(check)])
    assert output == b""

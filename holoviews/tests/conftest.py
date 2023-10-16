import contextlib

import pytest
from panel.tests.conftest import (  # noqa
    optional_markers,
    port,
    pytest_addoption,
    pytest_configure,
    server_cleanup,
)


def pytest_collection_modifyitems(config, items):
    skipped, selected = [], []
    markers = [m for m in optional_markers if config.getoption(f"--{m}")]
    empty = not markers
    for item in items:
        if empty and any(m in item.keywords for m in optional_markers):
            skipped.append(item)
        elif empty:
            selected.append(item)
        elif not empty and any(m in item.keywords for m in markers):
            selected.append(item)
        else:
            skipped.append(item)

    config.hook.pytest_deselected(items=skipped)
    items[:] = selected


with contextlib.suppress(ImportError):
    import matplotlib as mpl
    mpl.use('agg')


with contextlib.suppress(Exception):
    # From Dask 2023.7,1 they now automatic convert strings
    # https://docs.dask.org/en/stable/changelog.html#v2023-7-1
    import dask
    dask.config.set({"dataframe.convert-string": False})


@pytest.fixture
def ibis_sqlite_backend():
    try:
        import ibis
    except ImportError:
        yield None
    else:
        ibis.set_backend('sqlite')
        yield
        ibis.set_backend(None)

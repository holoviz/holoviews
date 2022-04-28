import pytest

from panel.io import state


@pytest.fixture(autouse=True)
def server_cleanup():
    """
    Clean up after test fails
    """
    try:
        yield
    finally:
        state.kill_all_servers()
        state._indicators.clear()
        state._locations.clear()
        state._curdoc = None
        state.cache.clear()

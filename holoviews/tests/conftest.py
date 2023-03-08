import asyncio

import pytest

from panel.io import state

# Tornado 6.2 has deprecated the use of IOLoop.current,
# when no asyncio event loop is running.
# This will start an event loop.
try:
    asyncio.get_running_loop()
except RuntimeError:
    # Create a new asyncio event loop for this thread.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


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
        state.cache.clear()

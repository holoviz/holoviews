from __future__ import annotations

from unittest.mock import patch

import pytest


class IPythonCase:
    """This class extends ComparisonTestCase to handle IPython specific
    objects and support the execution of cells and magic.

    """

    def setup_method(self):
        try:
            import IPython
        except Exception:
            pytest.skip("IPython could not be imported")

        from traitlets.config import Config

        with patch("atexit.register", lambda *args, **kwargs: None):
            config = Config()
            config.HistoryManager.enabled = False
            self.ip = IPython.InteractiveShell(config=config)

    def teardown_method(self) -> None:
        # self.ip.displayhook.flush calls gc.collect
        with patch("gc.collect", lambda: None):
            self.ip.reset(new_session=False)
        del self.ip

    def get_object(self, name):
        obj = self.ip._object_find(name).obj
        if obj is None:
            raise self.failureException(f"Could not find object {name}")
        return obj

    def cell(self, line):
        """Run an IPython cell"""
        self.ip.run_cell(line, silent=True)

    def cell_magic(self, *args, **kwargs):
        """Run an IPython cell magic"""
        self.ip.run_cell_magic(*args, **kwargs)

    def line_magic(self, *args, **kwargs):
        """Run an IPython line magic"""
        self.ip.run_line_magic(*args, **kwargs)

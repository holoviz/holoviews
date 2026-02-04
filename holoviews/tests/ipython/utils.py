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

        self.exits = []
        with patch('atexit.register', self.exits.append):
            config = Config()
            config.HistoryManager.hist_file = ':memory:'
            self.ip = IPython.InteractiveShell(
                config=config,
                history_length=0,
                history_load_length=0
            )

    def teardown_method(self) -> None:
        # Save references to database connections before exit callbacks
        # because exit callbacks set history_manager to None
        db_connections = []
        if hasattr(self.ip, 'history_manager') and self.ip.history_manager is not None:
            hm = self.ip.history_manager
            if hasattr(hm, 'db'):
                db_connections.append(hm.db)
            if hasattr(hm, 'save_thread') and hasattr(hm.save_thread, 'db'):
                db_connections.append(hm.save_thread.db)

        # self.ip.displayhook.flush calls gc.collect
        with patch('gc.collect', lambda: None):
            for ex in self.exits:
                ex()

            for db in db_connections:
                db.close()

        del self.ip

    def get_object(self, name):
        obj = self.ip._object_find(name).obj
        if obj is None:
            raise self.failureException(f"Could not find object {name}")
        return obj


    def cell(self, line):
        """Run an IPython cell

        """
        self.ip.run_cell(line, silent=True)

    def cell_magic(self, *args, **kwargs):
        """Run an IPython cell magic

        """
        self.ip.run_cell_magic(*args, **kwargs)


    def line_magic(self, *args, **kwargs):
        """Run an IPython line magic

        """
        self.ip.run_line_magic(*args, **kwargs)

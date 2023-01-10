"""
Unit tests relating to notebook processing
"""
import os

import nbconvert
import nbformat

from holoviews.element.comparison import ComparisonTestCase
from holoviews.ipython.preprocessors import OptsMagicProcessor, OutputMagicProcessor


def apply_preprocessors(preprocessors, nbname):
    notebooks_path = os.path.join(os.path.split(__file__)[0], 'notebooks')
    with open(os.path.join(notebooks_path, nbname)) as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)
        exporter = nbconvert.PythonExporter()
        for preprocessor in preprocessors:
            exporter.register_preprocessor(preprocessor)
        source, meta = exporter.from_notebook_node(nb)
    return source

class TestOptsPreprocessor(ComparisonTestCase):

    def test_opts_image_line_magic(self):
        nbname = 'test_opts_image_line_magic.ipynb'
        expected = """hv.util.opts(" Image [xaxis=None] (cmap='viridis')")"""
        source = apply_preprocessors([OptsMagicProcessor()], nbname)
        self.assertEqual(source.strip().endswith(expected), True)

    def test_opts_image_cell_magic(self):
        nbname = 'test_opts_image_cell_magic.ipynb'
        expected = ("""hv.util.opts(" Image [xaxis=None] (cmap='viridis')", """
                    + """hv.Image(np.random.rand(20,20)))""")
        source = apply_preprocessors([OptsMagicProcessor()], nbname)
        self.assertEqual(source.strip().endswith(expected), True)

    def test_opts_image_cell_magic_offset(self):
        nbname = 'test_opts_image_cell_magic_offset.ipynb'
        # FIXME: Not quite right yet, shouldn't have a leading space or a newline
        expected = (" 'An expression (literal) on the same line';\n"
                    + """hv.util.opts(" Image [xaxis=None] (cmap='viridis')", """
                    + """hv.Image(np.random.rand(20,20)))""")
        source = apply_preprocessors([OptsMagicProcessor()], nbname)
        self.assertEqual(source.strip().endswith(expected), False)

    def test_opts_image_line_magic_svg(self):
        nbname = 'test_output_svg_line_magic.ipynb'
        expected = """hv.util.output(" fig='svg'")"""
        source = apply_preprocessors([OutputMagicProcessor()], nbname)
        self.assertEqual(source.strip().endswith(expected), True)

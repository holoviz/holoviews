"""
python -m holoviews.util.command Conversion_Example.ipynb
OR
holoviews Conversion_Example.ipynb
"""

from __future__ import absolute_import

import sys, os
import nbformat, nbconvert
from ..ipython.preprocessors import OptsMagicProcessor, OutputMagicProcessor
from ..ipython.preprocessors import StripMagicsProcessor


def main(preprocessors=[OptsMagicProcessor(),
                                  OutputMagicProcessor(),
                                  StripMagicsProcessor()]):
    with open(sys.argv[1]) as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)
        exporter = nbconvert.PythonExporter()
        for preprocessor in preprocessors:
            exporter.register_preprocessor(preprocessor)
        source, meta = exporter.from_notebook_node(nb)
        print(source)


if __name__ == '__main__':
    main()


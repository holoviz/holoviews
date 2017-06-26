#! /usr/bin/env python
"""
python -m holoviews.util.command Conversion_Example.ipynb
OR
holoviews Conversion_Example.ipynb
"""

from __future__ import absolute_import, print_function

import sys, os
import nbformat, nbconvert
try:
    from ..ipython.preprocessors import OptsMagicProcessor, OutputMagicProcessor
    from ..ipython.preprocessors import StripMagicsProcessor
except:
    from holoviews.ipython.preprocessors import OptsMagicProcessor, OutputMagicProcessor
    from holoviews.ipython.preprocessors import StripMagicsProcessor

def main(filename=None,
         preprocessors=[OptsMagicProcessor(),
                        OutputMagicProcessor(),
                        StripMagicsProcessor()]):

    filename = filename if filename else sys.argv[1]
    with open(filename) as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)
        exporter = nbconvert.PythonExporter()
        for preprocessor in preprocessors:
            exporter.register_preprocessor(preprocessor)
        source, meta = exporter.from_notebook_node(nb)
        return source


if __name__ == '__main__':
    print(main(), file=sys.stdout)


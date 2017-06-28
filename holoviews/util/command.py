#! /usr/bin/env python
"""
python -m holoviews.util.command Conversion_Example.ipynb
OR
holoviews Conversion_Example.ipynb
"""

from __future__ import absolute_import, print_function

import sys
import os
import argparse
from argparse import RawTextHelpFormatter

try:
    import nbformat, nbconvert
except:
    print('Both nbformat and nbconvert need to be installed to use the holoviews command')
    sys.exit()
try:
    from ..ipython.preprocessors import OptsMagicProcessor, OutputMagicProcessor
    from ..ipython.preprocessors import StripMagicsProcessor
except:
    from holoviews.ipython.preprocessors import OptsMagicProcessor, OutputMagicProcessor
    from holoviews.ipython.preprocessors import StripMagicsProcessor


def main():
    if len(sys.argv) < 2:
        print("For help with the holoviews command run:\n\nholoviews --help\n")
        sys.exit()

    parser = argparse.ArgumentParser(prog='holoviews',
                                     formatter_class=RawTextHelpFormatter,
                                     description=description,
                                     epilog=epilog)

    parser.add_argument('notebook', metavar='notebook', type=str, nargs=1,
                    help='The Jupyter notebook to convert to Python syntax.')

    args = parser.parse_args()
    print(export_to_python(args.notebook[0]), file=sys.stdout)


def export_to_python(filename=None,
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



description = """
Command line interface for holoviews.

This utility allows conversion of notebooks containing the HoloViews
%opts, %%opts, %output and %%output magics to regular Python
syntax. This is useful for turning Jupyter notebooks using HoloViews
into Bokeh applications that can be served with:

bokeh server --show converted_notebook.py

The holoviews command supports the following options:
"""

epilog="""
Example usage
-------------

$ holoviews ./examples/demos/matplotlib/area_chart.ipynb

The converted syntax is then output to standard output where you can
direct it to a Python file of your choosing.
"""


if __name__ == '__main__':
    main()

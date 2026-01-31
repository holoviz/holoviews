import argparse
import os
import sys
from argparse import RawTextHelpFormatter

from . import examples


def main():
    if len(sys.argv) < 2:
        print("For help with the holoviews command run:\n\nholoviews --help\n")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        prog='holoviews',
        formatter_class=RawTextHelpFormatter,
        description=description,
    )

    parser.add_argument('--install-examples', metavar='install_examples',
                        type=str, nargs='?',
                        help='Install examples to the specified directory.')

    args = parser.parse_args()

    if args.install_examples is None:
        examples_dir = 'holoviews-examples'
    else:
        examples_dir = args.install_examples
    curdir = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join('..','..', curdir))
    examples(path=examples_dir, root=root)



description = """
Command line interface for holoviews.

The holoviews command supports the following options:
"""

if __name__ == '__main__':
    main()

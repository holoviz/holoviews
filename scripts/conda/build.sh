#!/usr/bin/env bash

set -euxo pipefail

PACKAGE="holoviews"

python -m build . # Can add -w when this is solved: https://github.com/pypa/hatch/issues/1305

VERSION=$(python -c "import $PACKAGE; print($PACKAGE._version.__version__)")
export VERSION

conda config --env --set conda_build.pkg_format 2
conda build scripts/conda/recipe --no-anaconda-upload --no-verify

mv "$CONDA_PREFIX/conda-bld/noarch/$PACKAGE-$VERSION-py_0.conda" dist

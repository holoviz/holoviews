#!/usr/bin/env bash

set -euxo pipefail

git status

python -m build -w .

git diff --exit-code

VERSION=$(find dist -name "*.whl" -exec basename {} \; | cut -d- -f2)
export VERSION
conda build scripts/conda/recipe --no-anaconda-upload --no-verify

#!/usr/bin/env bash

set -euxo pipefail

git status

export SETUPTOOLS_ENABLE_FEATURES="legacy-editable"
python -m build .

git diff --exit-code

VERSION=$(find dist -name "*.whl" -exec basename {} \; | cut -d- -f2)
export VERSION
conda build conda.recipe/ --no-test --no-anaconda-upload --no-verify

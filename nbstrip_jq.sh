#!/bin/bash

# `nbstrip_jq.sh`:
# This is a git filters that strips out Jupyter notebook outputs and meta data.
# Execute the following lines in order to activate this filter:
# conda install --yes jq  # or `brew install jq` or `apt-get install jq`
# 
# git config filter.nbstrip_jq.clean './nbstrip_jq.sh'
# git config filter.nbstrip_jq.smudge cat
# git config filter.nbstrip_jq.required true
# 
# Add the following line to your `.gitattributes`.
# *.ipynb filter=nbstrip_jq


jq --indent 1 \
    '
    (.cells[] | select(has("outputs")) | .outputs) = []
    | (.cells[] | select(has("execution_count")) | .execution_count) = null
    | .metadata = {}
    | .cells[].metadata = {}
    '

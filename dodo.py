import os
if "PYCTDEV_ECOSYSTEM" not in os.environ:
    os.environ["PYCTDEV_ECOSYSTEM"] = "conda"

from pyctdev import *  # noqa: api


def task_pip_on_conda():
    """Experimental: provide pip build env via conda"""
    return {'actions':[
        # some ecosystem=pip build tools must be installed with conda when using conda...
        'conda install -y pip twine wheel',
        # ..and some are only available via conda-forge
        'conda install -y -c conda-forge tox virtualenv',
        # this interferes with pip-installed nose
        'conda remove -y --force nose'
    ]}

import pyctdev._conda
python_develop = 'python -m pip install --no-deps --no-build-isolation -e .'
pyctdev._conda.python_develop = python_develop

from pyctdev._conda import _join_the_club, get_buildreqs


def _conda_build_deps(channel):
    buildreqs = get_buildreqs()
    deps = " ".join('"%s"'%_join_the_club(dep) for dep in buildreqs)
    if len(buildreqs)>0:
        return "mamba install -y %s %s"%(" ".join(['-c %s'%c for c in channel]),deps)
    else:
        return echo("Skipping conda install (no build dependencies)")

pyctdev._conda._conda_build_deps = _conda_build_deps

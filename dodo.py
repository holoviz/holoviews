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

from pyctdev._conda import _join_the_club, get_buildreqs, _get_dependencies, echo, _pin


def _conda_build_deps(channel):
    buildreqs = get_buildreqs()
    deps = " ".join('"%s"'%_join_the_club(dep) for dep in buildreqs)
    if len(buildreqs)>0:
        return "mamba install -y %s %s"%(" ".join(['-c %s'%c for c in channel]),deps)
    else:
        return echo("Skipping conda install (no build dependencies)")


def _conda_install_with_options(options,channel,env_name_again,no_pin_deps,all_extras):
    # TODO: list v string form for _pin
    deps = _get_dependencies(['install_requires']+options,all_extras=all_extras)
    deps = [_join_the_club(d) for d in deps]

    if len(deps)>0:
        deps = _pin(deps) if no_pin_deps is False else deps
        deps = " ".join('"%s"'%dep for dep in deps)
        # TODO and join the club?
        e = '' if env_name_again=='' else '-n %s'%env_name_again
        return "mamba install -y " + e + " %s %s"%(" ".join(['-c %s'%c for c in channel]),deps)
    else:
        return echo("Skipping conda install (no dependencies)")


pyctdev._conda._conda_build_deps = _conda_build_deps
pyctdev._conda._conda_install_with_options = _conda_install_with_options

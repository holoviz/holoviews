#!/usr/bin/env python

import json
import os
import sys
import shutil

from setuptools import setup, find_packages

import pyct.build

setup_args = {}
install_requires = [
    "param >=1.12.0,<3.0",
    "numpy >=1.0",
    "pyviz_comms >=0.7.4",
    "panel >=0.13.1",
    "colorcet",
    "packaging",
    "pandas >=0.20.0",
]

extras_require = {}

extras_require['lint'] = [
    'ruff',
    'pre-commit',
]

# Test requirements
extras_require['tests_core'] = [
    'pytest',
    'pytest-cov',
    'pytest-xdist',
    'flaky',
    'matplotlib >=3, <3.8',  # 3.8 breaks tests
    'nbconvert',
    'bokeh',
    'pillow',
    'plotly >=4.0',
    'dash >=1.16',
    'ipython >=5.4.0',
]

# Optional tests dependencies, i.e. one should be able
# to run and pass the test suite without installing any
# of those.
extras_require['tests'] = extras_require['tests_core'] + [
    'dask',
    'ibis-framework',  # Mapped to ibis-sqlite in setup.cfg for conda
    'xarray >=0.10.4',
    'networkx',
    'shapely',
    'ffmpeg',
    'cftime',
    'scipy',
    'selenium',
    'spatialpandas',
    'datashader >=0.11.1',
]

extras_require['test_ci'] = [
    'codecov',
    "pytest-github-actions-annotate-failures",
]

extras_require['tests_gpu'] = extras_require['tests'] + [
    'cudf',
]

extras_require['tests_nb'] = ['nbval']
extras_require['ui'] = ['playwright', 'pytest-playwright']

# Notebook dependencies
extras_require["notebook"] = ["ipython >=5.4.0", "notebook"]

# IPython Notebook + pandas + matplotlib + bokeh
extras_require["recommended"] = extras_require["notebook"] + [
    "matplotlib >=3",
    "bokeh >=2.4.3",
]

# Requirements to run all examples
extras_require["examples"] = extras_require["recommended"] + [
    "networkx",
    "pillow",
    "xarray >=0.10.4",
    "plotly >=4.0",
    'dash >=1.16',
    "streamz >=0.5.0",
    "ffmpeg",
    "cftime",
    "netcdf4",
    "dask",
    "scipy",
    "shapely",
    "scikit-image",
    "pyarrow",
    "pooch",
    "datashader >=0.11.1",
]


extras_require["examples_tests"] = extras_require["examples"] + extras_require['tests_nb']

# Extra third-party libraries
extras_require["extras"] = extras_require["examples"] + [
    "pscript ==0.7.1",
]

# Not used in tox.ini or elsewhere, kept for backwards compatibility.
extras_require["unit_tests"] = extras_require["examples"] + extras_require["tests"] + extras_require['lint']

extras_require['doc'] = extras_require['examples'] + [
    'nbsite >=0.8.2,<0.9.0',
    'mpl_sample_data >=3.1.3',
    'pscript',
    'graphviz',
    'bokeh >2.2',
    'pooch',
    'selenium',
]

extras_require['all'] = sorted(set(sum(extras_require.values(), [])))

extras_require['bokeh2'] = ["panel ==0.14.4", "param ==1.13.0"]  # Hard-pin to not pull in rc releases
extras_require['bokeh3'] = ["panel >=1.0.0"]

extras_require["build"] = [
    "param >=1.7.0",
    "setuptools >=30.3.0",
    "pyct >=0.4.4",
]

def get_setup_version(reponame):
    """
    Helper to get the current version from either git describe or the
    .version file (if available).
    """
    basepath = os.path.split(__file__)[0]
    version_file_path = os.path.join(basepath, reponame, ".version")
    try:
        from param import version
    except ImportError:
        version = None
    if version is not None:
        return version.Version.setup_version(
            basepath, reponame, archive_commit="$Format:%h$"
        )
    else:
        print(
            "WARNING: param>=1.6.0 unavailable. If you are installing a package, this warning can safely be ignored. If you are creating a package or otherwise operating in a git repository, you should install param>=1.6.0."
        )
        return json.load(open(version_file_path))["version_string"]


setup_args.update(
    dict(
        name="holoviews",
        version=get_setup_version("holoviews"),
        python_requires=">=3.8",
        install_requires=install_requires,
        extras_require=extras_require,
        description="Stop plotting your data - annotate your data and let it visualize itself.",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        author="Jean-Luc Stevens and Philipp Rudiger",
        author_email="holoviews@gmail.com",
        maintainer="HoloViz Developers",
        maintainer_email="developers@pyviz.org",
        platforms=["Windows", "Mac OS X", "Linux"],
        license="BSD",
        url="https://www.holoviews.org",
        project_urls={
            "Source": "https://github.com/holoviz/holoviews",
        },
        entry_points={"console_scripts": ["holoviews = holoviews.util.command:main"]},
        packages=find_packages(),
        include_package_data=True,
        classifiers=[
            "License :: OSI Approved :: BSD License",
            "Development Status :: 5 - Production/Stable",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Operating System :: OS Independent",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Natural Language :: English",
            "Framework :: Matplotlib",
            "Topic :: Scientific/Engineering",
            "Topic :: Software Development :: Libraries",
        ],
    )
)


if __name__ == "__main__":
    example_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "holoviews/examples"
    )

    if "develop" not in sys.argv and "egg_info" not in sys.argv:
        pyct.build.examples(example_path, __file__, force=True)

    if "install" in sys.argv:
        header = "HOLOVIEWS INSTALLATION INFORMATION"
        bars = "=" * len(header)

        extras = "\n".join("holoviews[%s]" % e for e in setup_args["extras_require"])

        print("%s\n%s\n%s" % (bars, header, bars))

        print("\nHoloViews supports the following installation types:\n")
        print("%s\n" % extras)
        print("Users should consider using one of these options.\n")
        print("By default only a core installation is performed and ")
        print("only the minimal set of dependencies are fetched.\n\n")
        print("For more information please visit http://holoviews.org/install.html\n")
        print(bars + "\n")

    setup(**setup_args)

    if os.path.isdir(example_path):
        shutil.rmtree(example_path)

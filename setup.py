#!/usr/bin/env python

import json
import os
import sys
import shutil

from setuptools import setup, find_packages

import pyct.build

setup_args = {}
install_requires = [
    "param >=1.9.3,<2.0",
    "numpy >=1.0",
    "pyviz_comms >=0.7.4",
    "panel >=0.8.0",
    "colorcet",
    "packaging",
    "pandas >=0.20.0",
]

extras_require = {}

# Notebook dependencies
extras_require["notebook"] = ["ipython >=5.4.0", "notebook"]

# IPython Notebook + pandas + matplotlib + bokeh
extras_require["recommended"] = extras_require["notebook"] + [
    "matplotlib >=3",
    "bokeh >=1.1.0",
]

# Requirements to run all examples
extras_require["examples"] = extras_require["recommended"] + [
    "networkx",
    "pillow",
    "xarray >=0.10.4",
    "plotly >=4.0",
    'dash >=1.16',
    "streamz >=0.5.0",
    "datashader >=0.11.1",
    "ffmpeg",
    "cftime",
    "netcdf4",
    "dask",
    "scipy",
    "shapely",
    "scikit-image"
]

# Extra third-party libraries
extras_require["extras"] = extras_require["examples"] + [
    "pscript ==0.7.1",
]

# Test requirements
extras_require['tests'] = [
    'nose',
    'mock',
    'flake8',
    'path.py',
    'matplotlib >=3',
    'nbsmoke >=0.2.0',
    'nbconvert',
    'codecov',
    # Numba currently incompatible with this Numpy version
    'numpy<1.22',
]

extras_require["unit_tests"] = extras_require["examples"] + extras_require["tests"]

if sys.version_info.major > 2:
    extras_require["unit_tests"].extend(
        [
            "pyarrow",
        ]  # spatialpandas incompatibility
    )

extras_require["basic_tests"] = (
    extras_require["tests"]
    + ["matplotlib >=3", "bokeh >=1.1.0", "pandas"]
    + extras_require["notebook"]
)

extras_require["nbtests"] = extras_require["recommended"] + [
    "nose",
    "deepdiff",
]

extras_require['doc'] = extras_require['examples'] + [
    'nbsite >=0.7.1',
    'sphinx',
    'mpl_sample_data >=3.1.3',
    'pscript',
    'graphviz',
    'bokeh >2.2',
    'pydata-sphinx-theme',
    'sphinx-copybutton',
    'pooch',
]

extras_require["build"] = [
    "param >=1.7.0",
    "setuptools >=30.3.0",
    "pyct >=0.4.4",
]

# Everything for examples and nosetests
extras_require["all"] = list(
    set(extras_require["unit_tests"]) | set(extras_require["nbtests"])
)

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
        return json.load(open(version_file_path, "r"))["version_string"]


setup_args.update(
    dict(
        name="holoviews",
        version=get_setup_version("holoviews"),
        python_requires=">=2.7",
        install_requires=install_requires,
        extras_require=extras_require,
        description="Stop plotting your data - annotate your data and let it visualize itself.",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        author="Jean-Luc Stevens and Philipp Rudiger",
        author_email="holoviews@gmail.com",
        maintainer="PyViz Developers",
        maintainer_email="developers@pyviz.org",
        platforms=["Windows", "Mac OS X", "Linux"],
        license="BSD",
        url="https://www.holoviews.org",
        entry_points={"console_scripts": ["holoviews = holoviews.util.command:main"]},
        packages=find_packages(),
        include_package_data=True,
        classifiers=[
            "License :: OSI Approved :: BSD License",
            "Development Status :: 5 - Production/Stable",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
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

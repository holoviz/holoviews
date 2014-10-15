#!/usr/bin/env python

import sys
from distutils.core import setup

setup_args = {}

required = {'param':">=0.0.1",
            'numpy':">=1.0",
            'matplotlib':">=1.1"}

packages_to_install = [required]
packages_to_state = [required]


if 'setuptools' in sys.modules:
    # support easy_install without depending on setuptools
    install_requires = []
    for package_list in packages_to_install:
        install_requires+=["%s%s"%(package,version) for package,version in package_list.items()]
    setup_args['install_requires']=install_requires
    setup_args['dependency_links']=["http://pypi.python.org/simple/"]
    setup_args['zip_safe'] = False

for package_list in packages_to_state:
    requires = []
    requires+=["%s (%s)"%(package,version) for package,version in package_list.items()]
    setup_args['requires']=requires


setup_args.update(dict(
    name='dataviews',
    version="0.7",
    description='Composable, declarative data structures for building complex visualizations in Python.',
    long_description=open('README.rst').read(),
    author= "Jean-Luc Stevens and Philipp Rudiger",
    author_email= "developers@topographica.org",
    maintainer= "IOAM",
    maintainer_email= "developers@topographica.org",
    platforms=['Windows', 'Mac OS X', 'Linux'],
    license='BSD',
    url='http://ioam.github.com/dataviews/',
    packages = ["dataviews",
                "dataviews.sheetviews",
                "dataviews.ipython",
                "dataviews.styles",
                "dataviews.plotting",
                "dataviews.interface"],
    package_data={'dataviews.styles': ['*.mplstyle']},
    classifiers = [
        "License :: OSI Approved :: BSD License",
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries"]
))


if __name__=="__main__":

    if 'upload' in sys.argv:
        import dataviews
        dataviews.__version__.verify(setup_args['version'])

    setup(**setup_args)

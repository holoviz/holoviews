#!/usr/bin/env python

import sys
from distutils.core import setup

setup_args = {}

required = {'param':">=1.3.1",
            'numpy':">=1.0",
            'matplotlib':">=1.3"}

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
    name='holoviews',
    version="1.0",
    description='Composable, declarative data structures for building complex visualizations easily.',
    long_description=open('README.rst').read(),
    author= "Jean-Luc Stevens and Philipp Rudiger",
    author_email= "developers@topographica.org",
    maintainer= "IOAM",
    maintainer_email= "developers@topographica.org",
    platforms=['Windows', 'Mac OS X', 'Linux'],
    license='BSD',
    url='http://ioam.github.com/holoviews/',
    packages = ["holoviews",
                "holoviews.core",
                "holoviews.element",
                "holoviews.interface",
                "holoviews.ipython",
                "holoviews.operation",
                "holoviews.plotting"],
    package_data={'holoviews.plotting': ['*.mplstyle'],
                  'holoviews.ipython': ['*.jinja']},
    classifiers = [
        "License :: OSI Approved :: BSD License",
        "Development Status :: 5 - Production/Stable",
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
        import holoviews
        holoviews.__version__.verify(setup_args['version'])

    setup(**setup_args)

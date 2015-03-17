#!/usr/bin/env python

import sys, os
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
    version="1.0.0",
    description='Composable, declarative data structures for building complex visualizations easily.',
    long_description=open('README.rst').read() if os.path.isfile('README.rst') else 'Consult README.rst',
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


def check_pseudo_package(path):
    """
    Verifies that a fake subpackage path for assets (notebooks, svgs,
    pngs etc) both exists and is populated with files.
    """
    if not os.path.isdir(path):
        raise Exception("Please make sure pseudo-package %s exists." % path)
    else:
        assets = os.listdir(path)
        if len(assets) == 0:
            raise Exception("Please make sure pseudo-package %s is populated." % path)


if __name__=="__main__":

    if ('upload' in sys.argv) or ('sdist' in sys.argv):
        # Make sure to create these directories and populate them before upload
        setup_args['packages'] += ["holoviews.assets", 'holoviews.notebooks']

        setup_args['package_data']['holoviews.assets'] = ['*.png', '*.rst']
        setup_args['package_data']['holoviews.notebooks'] = ['*.ipynb', '*.npy']

        check_pseudo_package(os.path.join('.', 'holoviews', 'assets'))
        check_pseudo_package(os.path.join('.', 'holoviews', 'notebooks'))

        import holoviews
        holoviews.__version__.verify(setup_args['version'])

    setup(**setup_args)

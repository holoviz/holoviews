#!/usr/bin/env python

import sys, os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup_args = {}
install_requires = ['param>=1.3.2', 'numpy>=1.0']
extras_require={}

# Notebook dependencies of IPython 3
extras_require['notebook-dependencies'] = ['ipython', 'pyzmq', 'jinja2', 'tornado',
                                           'jsonschema',  'notebook', 'pygments']
# IPython Notebook + matplotlib + Lancet
extras_require['recommended'] = (extras_require['notebook-dependencies']
                                 + ['matplotlib', 'lancet-ioam'])
# Additional, useful third-party packages
extras_require['extras'] = (['pandas', 'seaborn', 'mpld3', 'bokeh']
                            + extras_require['recommended'])
# Everything including cyordereddict (optimization) and nosetests
extras_require['all'] = (extras_require['recommended']
                         + extras_require['extras']
                         + ['cyordereddict', 'nose'])

setup_args.update(dict(
    name='holoviews',
    version="1.5.0",
    install_requires = install_requires,
    extras_require = extras_require,
    description='Stop plotting your data - annotate your data and let it visualize itself.',
    long_description=open('README.rst').read() if os.path.isfile('README.rst') else 'Consult README.rst',
    author= "Jean-Luc Stevens and Philipp Rudiger",
    author_email= "holoviews@gmail.com",
    maintainer= "IOAM",
    maintainer_email= "holoviews@gmail.com",
    platforms=['Windows', 'Mac OS X', 'Linux'],
    license='BSD',
    url='http://ioam.github.com/holoviews/',
    packages = ["holoviews",
                "holoviews.core",
                "holoviews.core.data",
                "holoviews.element",
                "holoviews.interface",
                "holoviews.ipython",
                "holoviews.operation",
                "holoviews.plotting",
                "holoviews.plotting.mpl",
                "holoviews.plotting.bokeh",
                "holoviews.plotting.widgets"],
    package_data={'holoviews.ipython': ['*.html'],
                  'holoviews.plotting.mpl': ['*.mplstyle', '*.jinja', '*.js'],
                  'holoviews.plotting.bokeh': ['*.js', '*.css'],
                  'holoviews.plotting.widgets': ['*.jinja', '*.js', '*.css']},
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

    if 'HOLOVIEWS_RELEASE' in os.environ:
        # Make sure to create these directories and populate them before upload
        setup_args['packages'] += ["holoviews.assets", 'holoviews.notebooks']

        # Add unit tests
        setup_args['packages'].append('holoviews.tests')

        setup_args['package_data']['holoviews.assets'] = ['*.png', '*.svg', '*.rst']
        setup_args['package_data']['holoviews.notebooks'] = ['*.ipynb', '*.npy']

        if ('upload' in sys.argv) or ('sdist' in sys.argv):
            check_pseudo_package(os.path.join('.', 'holoviews', 'tests'))
            check_pseudo_package(os.path.join('.', 'holoviews', 'assets'))
            check_pseudo_package(os.path.join('.', 'holoviews', 'notebooks'))

            import holoviews
            holoviews.__version__.verify(setup_args['version'])


    if 'install' in sys.argv:
        header = "HOLOVIEWS INSTALLATION INFORMATION"
        bars = "="*len(header)

        extras = '\n'.join('holoviews[%s]' % e for e in setup_args['extras_require'])

        print("%s\n%s\n%s" % (bars, header, bars))

        print("\nHoloViews supports the following installation types:\n")
        print("%s\n" % extras)
        print("Users should consider using one of these options.\n")
        print("By default only a core installation is performed and ")
        print("only the minimal set of dependencies are fetched.\n\n")
        print("For more information please visit http://holoviews.org/install.html\n")
        print(bars+'\n')


    setup(**setup_args)

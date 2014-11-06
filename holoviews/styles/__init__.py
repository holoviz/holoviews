"""
Defines global matplotlib styles and holoviews specific style options.

For setting a style see set_style.
"""

import os

import matplotlib.pyplot as plt
from matplotlib import rc_params_from_file

# Default Styles
from ..core.options import options, Cycle, StyleOpts, PlotOpts

options.Style = StyleOpts()

# Default Plotopts
options.Grid = PlotOpts()
options.GridLayout = PlotOpts()

options.Contours = StyleOpts(color='k')
options.Matrix = StyleOpts(cmap='gray', interpolation='nearest')
options.Array2D = StyleOpts(cmap='jet', interpolation='nearest')
options.HeatMap = StyleOpts(cmap='jet', interpolation='nearest')
options.GridLayout = StyleOpts(**{'font.size': 10, 'axes.labelsize': 'small',
                                  'axes.titlesize': 'small'})
# Color cycles can be removed once default style set and test data updated
options.Curve = StyleOpts(color=Cycle(), linewidth=2)
options.Scatter = StyleOpts(color=Cycle(), linewidth=2)
options.Annotation = StyleOpts()
options.Histogram = StyleOpts(ec='k', fc='w')
options.Table = StyleOpts()
options.Points = StyleOpts(color='r', marker='x')

# Defining the most common style options for holoviews
GrayNearest = StyleOpts(cmap='gray', interpolation='nearest')


styles = {'default': './default.mplstyle'}


def set_style(key):
    """
    Select a style by name, e.g. set_style('default'). To revert to the
    previous style use the key 'unset' or False.
    """
    if key is None:
        return
    elif not key or key in ['unset', 'backup']:
        if 'backup' in styles:
            plt.rcParams.update(styles['backup'])
        else:
            raise Exception('No style backed up to restore')
    elif key not in styles:
        raise KeyError('%r not in available styles.')
    else:
        path = os.path.join(os.path.dirname(__file__), styles[key])
        new_style = rc_params_from_file(path)
        styles['backup'] = dict(plt.rcParams)

        plt.rcParams.update(new_style)

"""
Defines global matplotlib styles and DataViews specific style options.

For setting a style see set_style.
"""

import matplotlib.pyplot as plt

from .options import options, Cycle, StyleOpts

# Default Styles
options.Style = StyleOpts()
options.Contours = StyleOpts(color='k')
options.SheetView = StyleOpts(cmap='gray', interpolation='nearest')
options.Matrix = StyleOpts(cmap='jet', interpolation='nearest')
options.HeatMap = StyleOpts(cmap='jet', interpolation='nearest')
# Color cycles can be removed once default style set and test data updated
options.Curve = StyleOpts(color=Cycle(), linewidth=2)
options.Scatter = StyleOpts(color=Cycle(), linewidth=2)
options.Annotation = StyleOpts()
options.Histogram = StyleOpts(ec='k', fc='w')
options.Table = StyleOpts()
options.Points = StyleOpts(color='r', marker='x')
options.VectorField = StyleOpts(cmap='jet')

# Defining the most common style options for dataviews
GrayNearest = StyleOpts(cmap='gray', interpolation='nearest')

# From https://gist.github.com/huyng/816622
default_style = {
    'axes.axisbelow': True,
     'axes.color_cycle': ['#348ABD',
      '#7A68A6',
      '#A60628',
      '#467821',
      '#CF4457',
      '#188487',
      '#E24A33'],
     'axes.edgecolor': '#bcbcbc',
     'axes.facecolor': '#eeeeee',
     'axes.grid': True,
     'axes.labelcolor': '#555555',
     'axes.labelsize': 'large',
     'axes.linewidth': 1.0,
     'axes.titlesize': 'x-large',
     'figure.edgecolor': 'white',
     'figure.facecolor': 'white',
     'figure.figsize': (6.0, 4.0),
     'figure.subplot.hspace': 0.5,
     'font.family': 'monospace',
     'font.monospace': ['Andale Mono',
      'Nimbus Mono L',
      'Courier New',
      'Courier',
      'Fixed',
      'Terminal',
      'monospace'],
     'font.size': 10,
     'interactive': True,
     'keymap.all_axes': ['a'],
     'keymap.back': ['left', 'c', 'backspace'],
     'keymap.forward': ['right', 'v'],
     'keymap.fullscreen': ['f'],
     'keymap.grid': ['g'],
     'keymap.home': ['h', 'r', 'home'],
     'keymap.pan': ['p'],
     'keymap.save': ['s'],
     'keymap.xscale': ['L', 'k'],
     'keymap.yscale': ['l'],
     'keymap.zoom': ['o'],
     'legend.fancybox': True,
     'lines.antialiased': True,
     'lines.linewidth': 1.0,
     'patch.antialiased': True,
     'patch.edgecolor': '#EEEEEE',
     'patch.facecolor': '#348ABD',
     'patch.linewidth': 0.5,
     'toolbar': 'toolbar2',
     'xtick.color': '#555555',
     'xtick.direction': 'in',
     'xtick.major.pad': 6.0,
     'xtick.major.size': 0.0,
     'xtick.minor.pad': 6.0,
     'xtick.minor.size': 0.0,
     'ytick.color': '#555555',
     'ytick.direction': 'in',
     'ytick.major.pad': 6.0,
     'ytick.major.size': 0.0,
     'ytick.minor.pad': 6.0,
     'ytick.minor.size': 0.0
}


dark_style = {
    'lines.color': 'white',
    'patch.edgecolor': 'white',
    'text.color': 'white',
    'axes.facecolor': 'black',
    'axes.edgecolor': 'white',
    'axes.labelcolor': 'white',
    'axes.color_cycle': ['8dd3c7', 'feffb3', 'bfbbd9',
                         'fa8174', '81b1d2', 'fdb462',
                         'b3de69', 'bc82bd', 'ccebc4', 'ffed6f'],
    'xtick.color': 'white',
    'ytick.color': 'white',
    'grid.color': 'white',
    'figure.facecolor': 'black',
    'figure.edgecolor': 'black',
    'savefig.facecolor': 'black',
    'savefig.edgecolor': 'black'}


ggplot_style = {
    'patch.linewidth': 0.5,
    'patch.facecolor': '#348ABD',
    'patch.edgecolor': '#eeeeee',
    'patch.antialiased': True,
    'font.size': 13.0,
    'axes.facecolor': '#bcbcbc',
    'axes.edgecolor': '#eeeeee',
    'axes.linewidth': 1,
    'axes.grid': True,
    'axes.titlesize': 'x-large',
    'axes.labelsize': 'large',
    'axes.labelcolor': '#555555',
    'axes.axisbelow': True,
    'axes.color_cycle': ['#E24A33', '#348ABD', '#988ED5', '#777777',
                         '#FBC15E', '#8EBA42', '#FFB5B8', '#0096fd',
                         '#004a7e', '#ff6805', '#d48c00', '#00d406',
                         '#006203', '#20e6f9', '#a020f9', '#f920c3'],
    'xtick.color': '#555555',
    'xtick.direction': 'out',
    'ytick.color': '#555555',
    'ytick.direction': 'out',
    'grid.color': 'white',
    'grid.linestyle': '-',
    'figure.facecolor': 'white',
    'figure.edgecolor': "gray"
}


grayscale_style = {
    'lines.color': 'black',
    'patch.facecolor': 'gray',
    'patch.edgecolor': 'black',
    'text.color': 'black',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',

    # black to light 'gray'
    'axes.color_cycle': [0.00, 0.40, 0.60, 0.70],
    'xtick.color': 'black',
    'ytick.color': 'black',
    'grid.color': 'black',
    'figure.facecolor': 0.75,
    'figure.edgecolor': 'white',
    'image.cmap': 'gray',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white'
}


styles = {'default': default_style,
          'dark': dark_style,
          'ggplot': ggplot_style,
          'grayscale': grayscale_style}


def set_style(key):
    """
    Select a style by name, e.g. set_style('ggplot'). To revert to the
    previous style use the key 'unset' or False.
    """
    if key is None:
        return
    elif not key or key in ['unset', 'backup']:
        if 'backup' in styles:
            plt.rcParams.update(styles['backup'])
    elif key not in styles:
        raise KeyError('%r not in available styles.')
    else:
        new_style = styles[key]
        styles['backup'] = dict([(k, plt.rcParams[k]) for k in new_style.keys()])

        plt.rcParams.update(new_style)

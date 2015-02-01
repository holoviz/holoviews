import matplotlib.pyplot as plt

from ..styles import set_style
from . import magics
from .magics import ViewMagic, load_magics
from .display_hooks import animate, set_display_hooks

from param import ipython as param_ext

try:    from matplotlib import animation
except: animation = None

all_line_magics = sorted(['%params', '%opts', '%view', '%channels'])
all_cell_magics = sorted(['%%view', '%%opts', '%%labels'])
message = """Welcome to the holoviews IPython extension! (http://ioam.github.io/holoviews/)"""
message += '\nAvailable magics: %s' % ', '.join(sorted(all_line_magics)
                                                + sorted(all_cell_magics))


def supported_formats(optional_formats):
    "Optional formats that are actually supported"
    supported = []
    for fmt in optional_formats:
        try:
            anim = animation.FuncAnimation(plt.figure(),
                                           lambda x: x, frames=[0,1])
            animate(anim, *ViewMagic.ANIMATION_OPTS[fmt])
            supported.append(fmt)
        except: pass
    return supported


def update_matplotlib_rc():
    """
    Default changes to the matplotlib rc used by IPython Notebook.
    """
    import matplotlib
    rc= {'figure.figsize': (6.0,4.0),
         'figure.facecolor': 'white',
         'figure.edgecolor': 'white',
         'font.size': 10,
         'savefig.dpi': 72,
         'figure.subplot.bottom' : .125
         }
    matplotlib.rcParams.update(rc)


_loaded = False
def load_ipython_extension(ip, verbose=True):

    if verbose: print(message)

    global _loaded
    if not _loaded:
        _loaded = True

        param_ext.load_ipython_extension(ip, verbose=False)

        load_magics(ip)
        valid_formats = supported_formats(ViewMagic.optional_formats)
        ViewMagic.register_supported_formats(valid_formats)
        set_display_hooks(ip)
        update_matplotlib_rc()
        set_style('default')

def unload_ipython_extension(ip):
    global _loaded
    _loaded = False

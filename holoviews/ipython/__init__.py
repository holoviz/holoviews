import matplotlib.pyplot as plt

from ..styles import set_style
from . import magics
from .magics import ViewMagic, load_magics
from .display_hooks import animate, set_display_hooks

from param import ipython as param_ext

try:    from matplotlib import animation
except: animation = None



all_line_magics = sorted(['%params', '%opts', '%view'])
all_cell_magics = sorted(['%%view', '%%opts', '%%labels'])
message = """Welcome to the holoviews IPython extension! (http://ioam.github.io/imagen/)"""
message += '\nAvailable magics: %s' % ', '.join(all_line_magics + all_cell_magics)


def select_format(format_priority):
    for fmt in format_priority:
        try:
            anim = animation.FuncAnimation(plt.figure(),
                                           lambda x: x, frames=[0,1])
            animate(anim, *magics.ANIMATION_OPTS[fmt])
            return fmt
        except: pass
    return format_priority[-1]


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


ViewMagic.VIDEO_FORMAT = select_format(['webm','h264','gif'])
# HTML_video output by default, but may be set to first_frame,
# middle_frame or last_frame (e.g. for testing purposes)

_loaded = False
def load_ipython_extension(ip, verbose=True):

    if verbose: print(message)

    global _loaded
    if not _loaded:
        _loaded = True

        param_ext.load_ipython_extension(ip, verbose=False)

        load_magics(ip)
        set_display_hooks(ip)
        update_matplotlib_rc()
        set_style('default')

def unload_ipython_extension(ip):
    global _loaded
    _loaded = False

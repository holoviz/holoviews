import os

import param

from ..core.options import Cycle, Options
from . import seaborn # pyflakes:ignore (API import)
from . import pandas # pyflakes:ignore (API import)
from .annotation import * # pyflakes:ignore (API import)
from .chart import * # pyflakes:ignore (API import)
from .plot import * # pyflakes:ignore (API import)
from .raster import * # pyflakes:ignore (API import)
from .tabular import * # pyflakes:ignore (API import)


class PlotSaver(param.ParameterizedFunction):
    """
    Parameterized function for saving the plot of a ViewableElement object to
    disk either as a static figure or as an animation. Keywords that
    are not parameters are passed into the anim method of the
    appropriate plot for animations and into matplotlib.figure.savefig
    for static figures.
    """

    size = param.NumericTuple(default=(5, 5), doc="""
      The matplotlib figure size in inches.""")

    filename = param.String(default=None, allow_None=True, doc="""
      This is the filename of the saved file. The output file type
      will be inferred from the file extension.""")

    anim_opts = param.Dict(
        default={'.webm':({}, ['-vcodec', 'libvpx', '-b', '1000k']),
                 '.mp4':({'codec':'libx264'}, ['-pix_fmt', 'yuv420p']),
                 '.gif':({'fps':10}, [])}, doc="""
        The arguments to matplotlib.animation.save for different
        animation formats. The key is the file extension and the tuple
        consists of the kwargs followed by a list of arguments for the
        'extra_args' keyword argument.""" )


    def __call__(self, view, **params):
        anim_exts = ['.webm', '.mp4', 'gif']
        image_exts = ['.png', '.jpg', '.svg']
        writers = {'.mp4': 'ffmpeg', '.webm':'ffmpeg', '.gif':'imagemagick'}
        supported_extensions = image_exts + anim_exts

        self.p = param.ParamOverrides(self, params, allow_extra_keywords=True)
        if self.p.filename is None:
            raise Exception("Please supply a suitable filename.")

        _, ext = os.path.splitext(self.p.filename)
        if ext not in supported_extensions:
            valid_exts = ', '.join(supported_extensions)
            raise Exception("File of type %r not in %s" % (ext, valid_exts))
        file_format = ext[1:]

        plottype = Plot.defaults[type(view)]
        plotopts = self.lookup_options(view, 'plot').options
        plot = plottype(view, **dict(plotopts, size=self.p.size))

        if len(plot) > 1 and ext in anim_exts:
            anim_kwargs, extra_args = self.p.anim_opts[ext]
            anim = plot.anim(**self.p.extra_keywords())
            anim.save(self.p.filename, writer=writers[ext],
                      **dict(anim_kwargs, extra_args=extra_args))
        elif len(plot) > 1:
            raise Exception("Invalid extension %r for animation." % ext)
        elif ext in anim_exts:
            raise Exception("Invalid extension %r for figure." % ext)
        else:
            plot().savefig(filename=self.p.filename, format=file_format,
                           **self.p.extra_keywords())


Plot.register_options()
Plot.options.Contours = Options('style', color='k')
Plot.options.Matrix = Options('style', cmap='gray', interpolation='nearest')
Plot.options.Raster = Options('style', cmap='jet', interpolation='nearest')
Plot.options.HeatMap = Options('style', cmap='jet', interpolation='nearest')
Plot.options.GridLayout = Options('style', **{'font.size': 10, 'axes.labelsize': 'small',
                                                 'axes.titlesize': 'small'})
# Color cycles can be removed once default style set and test data updated
Plot.options.Curve = Options('style', color=Cycle(), linewidth=2)
Plot.options.Scatter = Options('style', color=Cycle(), marker='o')
Plot.options.Histogram = Options('style', ec='k', fc='w')
Plot.options.Points = Options('style', color=Cycle(), marker='o')

# Defining the most common style options for holoviews
GrayNearest = Options(key='style', cmap='gray', interpolation='nearest')

def public(obj):
    if not isinstance(obj, type): return False
    baseclasses = [Plot]
    return any([issubclass(obj, bc) for bc in baseclasses])


_public = ["PlotSaver", "GrayNearest"] + list(set([_k for _k, _v in locals().items() if public(_v)]))
__all__ = _public

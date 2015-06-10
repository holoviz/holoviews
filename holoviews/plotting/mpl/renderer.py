import os
import warnings
from io import BytesIO
from tempfile import NamedTemporaryFile

from ...core import HoloMap, AdjointLayout
from ...core.options import Store, StoreOptions

from .plot import MPLPlot
from .. import MIME_TYPES
from ..renderer import Renderer

from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.tight_bbox as tight_bbox
from matplotlib.transforms import Bbox, TransformedBbox, Affine2D

import param
from param.parameterized import bothmethod

# <format name> : (animation writer, format,  anim_kwargs, extra_args)
ANIMATION_OPTS = {
    'webm': ('ffmpeg', 'webm', {},
             ['-vcodec', 'libvpx', '-b', '1000k']),
    'mp4': ('ffmpeg', 'mp4', {'codec': 'libx264'},
             ['-pix_fmt', 'yuv420p']),
    'gif': ('imagemagick', 'gif', {'fps': 10}, []),
    'scrubber': ('html', None, {'fps': 5}, None)
}



def opts(el, percent_size):
    "Returns the plot options with supplied size (if not overridden)"
    obj = el.last if isinstance(el, HoloMap) else el
    options = MPLRenderer.get_plot_size(obj, percent_size) #  Store.registry[type(el)].renderer
    options.update(Store.lookup_options(obj, 'plot').options)
    return options


class MPLRenderer(Renderer):
    """
    Exporter used to render data from matplotlib, either to a stream
    or directly to file.

    The __call__ method renders an HoloViews component to raw data of
    a specified matplotlib format.  The save method is the
    corresponding method for saving a HoloViews objects to disk.

    The save_fig and save_anim methods are used to save matplotlib
    figure and animation objects. These match the two primary return
    types of plotting class implemented with matplotlib.
    """
    drawn = {}

    def __call__(self, obj, fmt=None):
        """
        Render the supplied HoloViews component using matplotlib.
        """
        if isinstance(obj, AdjointLayout):
            obj = Layout.from_values(obj)

        element_type = obj.type if isinstance(obj, HoloMap) else type(obj)
        try:
            plotclass = Store.registry[element_type]
        except KeyError:
            raise Exception("No corresponding plot type found for %r" % type(obj))

        plot = plotclass(obj, **opts(obj,  self.size))

        if fmt is None:
            fmt = self.holomap if len(plot) > 1 else self.fig
            if fmt is None: return

        if len(plot) > 1:
            (writer, _, anim_kwargs, extra_args) = ANIMATION_OPTS[fmt]
            anim = plot.anim(fps=self.fps)
            if extra_args != []:
                anim_kwargs = dict(anim_kwargs, extra_args=extra_args)

            data = self.anim_data(anim, fmt, writer, **anim_kwargs)
        else:
            data = self.figure_data(plot.update(0), fmt, **({'dpi':self.dpi} if self.dpi else {}))

        return data, {'file-ext':fmt,
                      'mime_type':MIME_TYPES[fmt]}

    @classmethod
    def get_plot_size(cls, obj, percent_size):
        """
        Given a holoviews object and a percentage size, apply heuristics
        to compute a suitable figure size. For instance, scaling layouts
        and grids linearly can result in unwieldy figure sizes when there
        are a large number of elements. As ad hoc heuristics are used,
        this functionality is kept separate from the plotting classes
        themselves.

        Used by the IPython Notebook display hooks and the save
        utility. Note that this can be overridden explicitly per object
        using the fig_size and size plot options.
        """
        factor = percent_size / 100.0
        plot_type = obj.type if isinstance(obj, HoloMap) else type(obj)
        options = Store.lookup_options(obj, 'plot').options
        fig_inches = options.get('fig_inches', MPLPlot.fig_inches)
        if isinstance(fig_inches, (list, tuple)):
            fig_inches =  (fig_inches[0] * factor,
                           fig_inches[1] * factor)
        else:
            fig_inches = MPLPlot.fig_inches * factor

        return dict(fig_inches=fig_inches)


    @bothmethod
    def supported_holomap_formats(self_or_cls, optional_formats):
        "Optional formats that are actually supported by this renderer"
        supported = []
        with param.logging_level('CRITICAL'):
            self_or_cls.HOLOMAP_FORMAT_ERROR_MESSAGES = {}
        for fmt in optional_formats:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fig = plt.figure()
                    anim = animation.FuncAnimation(fig, lambda x: x, frames=[0,1])
                    (writer, fmt, anim_kwargs, extra_args) = ANIMATION_OPTS[fmt]
                    if extra_args != []:
                        anim_kwargs = dict(anim_kwargs, extra_args=extra_args)
                        renderer = self_or_cls.instance(dpi=72)
                        renderer.anim_data(anim, fmt, writer, **anim_kwargs)
                    plt.close(fig)
                    supported.append(fmt)
                except Exception as e:
                    self_or_cls.HOLOMAP_FORMAT_ERROR_MESSAGES[fmt] = str(e)
        return supported


    @bothmethod
    def save(self_or_cls, obj, basename, fmt=None, key={}, info={}, options=None, **kwargs):
        """
        Save a HoloViews object to file, either using an explicitly
        supplied format or to the appropriate default.
        """
        if info or key:
            raise Exception('MPLRenderer does not support saving metadata to file.')

        with StoreOptions.options(obj, options, **kwargs):
            rendered = self_or_cls(obj, fmt)
        if rendered is None: return
        (data, info) = rendered
        filename ='%s.%s' % (basename, info['file-ext'])
        with open(filename, 'wb') as f:
            f.write(self_or_cls.encode(rendered))

    def anim_data(self, anim, fmt, writer, **anim_kwargs):
        """
        Render a matplotlib animation object and return the corresponding data.
        """
        anim_kwargs = dict(anim_kwargs, **({'dpi':self.dpi} if self.dpi is not None else {}))
        anim_kwargs = dict({'fps':self.fps} if fmt =='gif' else {}, **anim_kwargs)
        if not hasattr(anim, '_encoded_video'):
            with NamedTemporaryFile(suffix='.%s' % fmt) as f:
                anim.save(f.name, writer=writer,
                          **dict(anim_kwargs, **({'dpi':self.dpi} if self.dpi else {})))
                video = open(f.name, "rb").read()
        return video


    def _compute_bbox(self, fig, kw):
        """
        Compute the tight bounding box for each figure once, reducing
        number of required canvas draw calls from N*2 to N+1 as a
        function of the number of frames.

        Tight bounding box computing code here mirrors:
        matplotlib.backend_bases.FigureCanvasBase.print_figure
        as it hasn't been factored out as a function.
        """
        fig_id = id(fig)
        if kw['bbox_inches'] == 'tight' and kw['format'] == 'png':
            if not fig_id in MPLRenderer.drawn:
                fig.set_dpi(self.dpi)
                fig.canvas.draw()
                renderer = fig._cachedRenderer
                bbox_inches = fig.get_tightbbox(renderer)
                bbox_artists = fig.get_default_bbox_extra_artists()
                bbox_filtered = []
                for a in bbox_artists:
                    bbox = a.get_window_extent(renderer)
                    if a.get_clip_on():
                        clip_box = a.get_clip_box()
                        if clip_box is not None:
                            bbox = Bbox.intersection(bbox, clip_box)
                        clip_path = a.get_clip_path()
                        if clip_path is not None and bbox is not None:
                            clip_path = clip_path.get_fully_transformed_path()
                            bbox = Bbox.intersection(bbox,
                                                     clip_path.get_extents())
                    if bbox is not None and (bbox.width != 0 or
                                             bbox.height != 0):
                        bbox_filtered.append(bbox)
                if bbox_filtered:
                    _bbox = Bbox.union(bbox_filtered)
                    trans = Affine2D().scale(1.0 / self.dpi)
                    bbox_extra = TransformedBbox(_bbox, trans)
                    bbox_inches = Bbox.union([bbox_inches, bbox_extra])
                pad = plt.rcParams['savefig.pad_inches']
                bbox_inches = bbox_inches.padded(pad)
                MPLRenderer.drawn[fig_id] = bbox_inches
                kw['bbox_inches'] = bbox_inches
            else:
                kw['bbox_inches'] = MPLRenderer.drawn[fig_id]
        return kw


    def figure_data(self, fig, fmt='png', bbox_inches='tight', **kwargs):
        """
        Render matplotlib figure object and return the corresponding data.

        Similar to IPython.core.pylabtools.print_figure but without
        any IPython dependency.
        """
        kw = dict(
            format=fmt,
            facecolor=fig.get_facecolor(),
            edgecolor=fig.get_edgecolor(),
            dpi=self.dpi,
            bbox_inches=bbox_inches,
        )
        kw.update(kwargs)

        # Attempts to precompute the tight bounding box
        try:
            kw = self._compute_bbox(fig, kw)
        except:
            pass

        bytes_io = BytesIO()
        fig.canvas.print_figure(bytes_io, **kw)
        data = bytes_io.getvalue()
        if fmt == 'svg':
            data = data.decode('utf-8')
        return data


Store.renderer = MPLRenderer


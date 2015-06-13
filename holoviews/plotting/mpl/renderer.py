import os, sys, uuid
import warnings
from io import BytesIO
from tempfile import NamedTemporaryFile

from ...core import HoloMap, AdjointLayout
from ...core.options import Store, StoreOptions

from .. import MIME_TYPES
from ..plot import Plot
from ..renderer import Renderer

from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.tight_bbox as tight_bbox
from matplotlib.transforms import Bbox, TransformedBbox, Affine2D

from matplotlib.backends.backend_nbagg import CommSocket

import param
from param.parameterized import bothmethod


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

    backend = param.String('matplotlib', doc="The backend name.")

    # <format name> : (animation writer, format,  anim_kwargs, extra_args)
    ANIMATION_OPTS = {
        'webm': ('ffmpeg', 'webm', {},
                 ['-vcodec', 'libvpx', '-b', '1000k']),
        'mp4': ('ffmpeg', 'mp4', {'codec': 'libx264'},
                 ['-pix_fmt', 'yuv420p']),
        'gif': ('imagemagick', 'gif', {'fps': 10}, []),
        'scrubber': ('html', None, {'fps': 5}, None)
    }


    def __call__(self, obj, fmt=None):
        """
        Render the supplied HoloViews component using matplotlib.
        """
        if not isinstance(obj, Plot):
            obj = Layout.from_values(obj) if isinstance(obj, AdjointLayout) else obj
            element_type = obj.type if isinstance(obj, HoloMap) else type(obj)
            try:
                plotclass = Store.registry[element_type]
            except KeyError:
                raise Exception("No corresponding plot type found for %r" % type(obj))

            if fmt is None:
                fmt = self.holomap if len(plot) > 1 else self.fig
                if fmt is None: return

            plot = plotclass(obj, **self.plot_options(obj, self.size))
            plot.update(0)

        elif fmt is None:
            raise Exception("Format must be specified when supplying a plot instance")
        else:
            plot = obj

        if fmt in ['png', 'svg', 'pdf']:
            data = self._figure_data(plot, fmt, **({'dpi':self.dpi} if self.dpi else {}))
        else:

            if sys.version_info[0] == 3 and mpl.__version__[:-2] in ['1.2', '1.3']:
                raise Exception("<b>Python 3 matplotlib animation support broken &lt;= 1.3</b>")
            anim = plot.anim(fps=self.fps)
            data = self._anim_data(anim, fmt)

        return data, {'file-ext':fmt,
                      'mime_type':MIME_TYPES[fmt]}

    @classmethod
    def plot_options(cls, obj, percent_size):
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
        from .plot import MPLPlot
        factor = percent_size / 100.0
        obj = obj.last if isinstance(obj, HoloMap) else obj
        options = Store.lookup_options(obj, 'plot').options
        fig_inches = options.get('fig_inches', MPLPlot.fig_inches)
        if isinstance(fig_inches, (list, tuple)):
            fig_inches =  (fig_inches[0] * factor,
                           fig_inches[1] * factor)
        else:
            fig_inches = MPLPlot.fig_inches * factor

        return dict({'fig_inches':fig_inches},
                    **Store.lookup_options(obj, 'plot').options)

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


    @classmethod
    def get_figure_manager(cls, counter, plot):
        try:
            from matplotlib.backends.backend_nbagg import new_figure_manager_given_figure
            from mpl_toolkits.mplot3d import Axes3D
        except:
            return None
        fig = plot.state
        manager = new_figure_manager_given_figure(counter, fig)
        # Need to call mouse_init on each 3D axis to enable rotation support
        for ax in fig.get_axes():
            if isinstance(ax, Axes3D):
                ax.mouse_init()
        return manager


    @bothmethod
    def get_size(self_or_cls, plot):
        w, h = plot.state.get_size_inches()
        dpi = plot.state.dpi
        return (w*dpi, h*dpi)


    def _figure_data(self, plot, fmt='png', bbox_inches='tight', **kwargs):
        """
        Render matplotlib figure object and return the corresponding data.

        Similar to IPython.core.pylabtools.print_figure but without
        any IPython dependency.
        """
        fig = plot.state
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


    def _anim_data(self, anim, fmt):
        """
        Render a matplotlib animation object and return the corresponding data.
        """
        (writer, _, anim_kwargs, extra_args) = self.ANIMATION_OPTS[fmt]
        if extra_args != []:
            anim_kwargs = dict(anim_kwargs, extra_args=extra_args)
        if fmt=='gif':
            anim_kwargs['fps'] = fps

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



class WidgetCommSocket(CommSocket):
    """
    CustomCommSocket provides communication between the IPython
    kernel and a matplotlib canvas element in the notebook.
    A CustomCommSocket is required to delay communication
    between the kernel and the canvas element until the widget
    has been rendered in the notebook.
    """

    def __init__(self, manager):
        self.supports_binary = None
        self.manager = manager
        self.uuid = str(uuid.uuid4())
        self.html = "<div id=%r></div>" % self.uuid

    def start(self):
        from IPython.kernel.comm import Comm
        try:
            self.comm = Comm('matplotlib', data={'id': self.uuid})
        except AttributeError:
            raise RuntimeError('Unable to create an IPython notebook Comm '
                               'instance. Are you in the IPython notebook?')
        self.comm.on_msg(self.on_message)
        self.comm.on_close(lambda close_message: self.manager.clearup_closed())


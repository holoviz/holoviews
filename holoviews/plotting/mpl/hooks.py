import copy
import numpy as np
from matplotlib import pyplot as plt

try:
    from mpld3 import plugins
except:
    plugins = None

import param

from ...core import NdOverlay, Overlay
from ...element import HeatMap, Raster, Scatter, Curve, Points, Bars, Histogram
from . import CurvePlot, PointPlot, OverlayPlot, RasterPlot, HistogramPlot, BarPlot


class PlottingHook(param.ParameterizedFunction):
    """
    PlottingHooks can be used to extend the default functionality
    of HoloViews. Each ElementPlot type can be provided with a list of
    hooks to apply at the end of plotting. The PlottingHook is provided
    with ElementPlot instance, which gives access to the figure, axis
    and artists via the handles, and the Element currently displayed
    by the Plot. Since each Plot can be associated with multiple
    Element plot types the Element types are validated against the
    Elements in the types parameter.
    """

    types = param.List([], doc="""List of types processed by the hook.""")

    __abstract = True

    def _applies(self, plot, view):
        return type(view) in self.types


class MplD3Plugin(PlottingHook):
    """
    The mpld3 library available as an optional backend
    for HoloViews provides the option of adding
    interactivity to the plot through various plugins.
    Subclasses of this PlottingHook can enable
    """

    css = param.String(doc="""CSS applied to HTML mpld3 Plugins.""", default="""
          table {border-collapse: collapse;}
          th {color: #ffffff; background-color: #000000;}
          td {background-color: #cccccc;}
          table, th, td {font-family:Arial, Helvetica, sans-serif;
                         border: 1px solid black; text-align: right;}""")

    hoffset = param.Integer(default=10, doc="Vertical offset of the labels.")

    voffset = param.Integer(default=10, doc="Horizontal offset of the labels.")

    __abstract = True

    def _applies(self, plot, view):
        from ..ipython.magics import OutputMagic
        types_match = super(MplD3Plugin, self)._applies(plot, view)
        axes3d = plot.projection == '3d'
        mpld3_backend = OutputMagic.options['backend'] == 'd3'
        return types_match and mpld3_backend and not axes3d



class PointPlugin(MplD3Plugin):
    "Labels each point with a table of its values."

    types = param.List([Points, Scatter])

    def __call__(self, plot, view):
        if not self._applies(plot, view): return
        fig = plot.handles['fig']
        df = view.dframe()
        labels = []
        for i in range(len(df)):
            label = df.ix[[i], :].T
            label.columns = [view.label]
            labels.append(str(label.to_html(header=len(view.label)>0)))
        tooltip = plugins.PointHTMLTooltip(plot.handles['paths'], labels,
                                           voffset=self.voffset, hoffset=self.hoffset,
                                           css=self.css)
        plugins.connect(fig, tooltip)



class CurvePlugin(MplD3Plugin):
    "Labels each line with the Curve objects label"

    format_string = param.String(default='<h4>{label}</h4>', doc="""
       Defines the HTML representation of the Element label""")

    types = param.List([Curve])

    def __call__(self, plot, view):
        if not self._applies(plot, view): return
        fig = plot.handles['fig']
        labels = [self.format_string.format(label=view.label)]
        tooltip = plugins.LineHTMLTooltip(plot.handles['line_segment'], labels,
                                          voffset=self.voffset, hoffset=self.hoffset,
                                          css=self.css)
        plugins.connect(fig, tooltip)


class BarPlugin(MplD3Plugin):

    types = param.List([Bars])

    def __call__(self, plot, view):
        if not self._applies(plot, view): return
        fig = plot.handles['fig']

        for key, bar in plot.handles['bars'].items():
            handle = bar.get_children()[0]
            selection = [(d.name,{k}) for d, k in zip(plot.bar_dimensions, key)
                         if d is not None]
            label_data = view.select(**dict(selection)).dframe().ix[0].to_frame()
            label = str(label_data.to_html(header=len(view.label)>0))
            tooltip = plugins.LineHTMLTooltip(handle, label, voffset=self.voffset,
                                              hoffset=self.hoffset, css=self.css)
            plugins.connect(fig, tooltip)




class HistogramPlugin(MplD3Plugin):
    "Labels each bar with a table of its values."

    types = param.List([Histogram])

    def __call__(self, plot, view):
        if not self._applies(plot, view): return
        fig = plot.handles['fig']

        df = view.dframe()
        labels = []
        for i in range(len(df)):
            label = df.ix[[i], :].T
            label.columns = [view.label]
            labels.append(str(label.to_html(header=len(view.label)>0)))

        for i, (bar, label) in enumerate(zip(plot.handles['bars'].get_children(), labels)):
            tooltip = plugins.LineHTMLTooltip(bar, label, voffset=self.voffset,
                                              hoffset=self.hoffset, css=self.css)
            plugins.connect(fig, tooltip)



class RasterPlugin(MplD3Plugin):
    """
    Replaces the imshow based Raster image with a
    pcolormesh, allowing each pixel to be labelled.
    """

    types = param.List(default=[Raster, HeatMap])

    def __call__(self, plot, view):
        if not self._applies(plot, view): return

        fig = plot.handles['fig']
        ax = plot.handles['axis']
        valid_opts = ['cmap']

        opts = {k:v for k,v, in plot.style.options.items()
                if k in valid_opts}

        data = view.data
        rows, cols = view.data.shape
        if isinstance(view, HeatMap):
            data = np.ma.array(data, mask=np.isnan(data))
            cmap = copy.copy(plt.cm.get_cmap(opts.get('cmap', 'gray')))
            cmap.set_bad('w', 1.)
            opts['cmap'] = cmap
            df = view.dframe(True).fillna(0)
            df = df.sort([d.name for d in view.dimensions()[1:2]])[::-1]
            l, b, r, t = (0, 0, 1, 1)
            data = np.flipud(data)
        else:
            df = view.dframe().sort(['y','x'], ascending=(1,1))[::-1]
            l, b, r, t = (0, 0, cols, rows)

        for k, ann in plot.handles.get('annotations', {}).items():
            ann.remove()
            plot.handles['annotations'].pop(k)

        # Generate color mesh to label each point
        cols+=1; rows+=1
        cmin, cmax = view.range(2)
        x, y = np.meshgrid(np.linspace(l, r, cols), np.linspace(b, t, rows))
        plot.handles['im'].set_visible(False)
        mesh = ax.pcolormesh(x, y, data, vmin=cmin, vmax=cmax, **opts)
        ax.invert_yaxis() # Doesn't work uninverted
        df.index = range(len(df))
        labels = []
        for i in range(len(df)):
            label = df.ix[[i], :].T
            label.columns = [' '.join([view.label, view.group])]
            labels.append(str(label.to_html(header=len(view.label)>0)))

        tooltip = plugins.PointHTMLTooltip(mesh, labels[::-1], hoffset=self.hoffset,
                                           voffset=self.voffset, css=self.css)
        plugins.connect(fig, tooltip)



class LegendPlugin(MplD3Plugin):
    """
    Provides an interactive legend allowing selecting
    and unselecting of different elements.
    """

    alpha_unsel = param.Number(default=0.2, doc="""
       The alpha level of the unselected elements""")

    alpha_sel = param.Number(default=2.0, doc="""
       The alpha level of the unselected elements""")

    types = param.List([Overlay, NdOverlay])

    def __call__(self, plot, view):
        if not self._applies(plot, view): return
        fig = plot.handles['fig']
        if 'legend' in plot.handles:
            plot.handles['legend'].set_visible(False)
        line_segments, labels = [], []
        keys = view.keys()
        for idx, subplot in enumerate(plot.subplots.values()):
            if isinstance(subplot, PointPlot):
                line_segments.append(subplot.handles['paths'])
                if isinstance(view, NdOverlay):
                    labels.append(str(keys[idx]))
                else:
                    labels.append(subplot.hmap.last.label)
            elif isinstance(subplot, CurvePlot):
                line_segments.append(subplot.handles['line_segment'])
                if isinstance(view, NdOverlay):
                    labels.append(str(keys[idx]))
                else:
                    labels.append(subplot.hmap.last.label)

        tooltip = plugins.InteractiveLegendPlugin(line_segments, labels,
                                                  alpha_sel=self.alpha_sel,
                                                  alpha_unsel=self.alpha_unsel)
        plugins.connect(fig, tooltip)


if plugins is not None:
    OverlayPlot.finalize_hooks = [LegendPlugin]
    RasterPlot.finalize_hooks = [RasterPlugin]
    CurvePlot.finalize_hooks = [CurvePlugin]
    PointPlot.finalize_hooks = [PointPlugin]
    HistogramPlot.finalize_hooks = [HistogramPlugin]
    BarPlot.finalize_hooks = [BarPlugin]

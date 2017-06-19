from __future__ import absolute_import

import param
from bokeh.models import GlyphRenderer, ColumnDataSource, DataRange1d, Range1d

from ...core.options import abbreviated_exception, SkipRendering
from ...core.spaces import DynamicMap
from .element import LegendPlot, line_properties
from ..util import match_spec
from .util import update_plot, bokeh_version

try:
    if bokeh_version > '0.12.5':
        from bkcharts import BoxPlot as BokehBoxPlot
    else:
        from bokeh.charts import BoxPlot as BokehBoxPlot
except:
    BokehBoxPlot = None


class ChartPlot(LegendPlot):
    """
    ChartPlot creates and updates Bokeh high-level Chart instances.
    The current implementation requires creating a new Chart for each
    frame and updating the existing Chart. Once Bokeh supports updating
    Charts directly this workaround will no longer be required.
    """

    def initialize_plot(self, ranges=None, plot=None, plots=None, source=None):
        """
        Initializes a new plot object with the last available frame.
        """
        # Get element key and ranges for frame
        element = self.hmap.last
        key = self.keys[-1]
        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = match_spec(element, ranges)
        self.current_ranges = ranges
        self.current_frame = element
        self.current_key = key

        # Initialize plot, source and glyph
        if plot is not None:
            raise Exception("Can't overlay Bokeh Charts based plot properties")

        init_element = element.clone(element.interface.concat(self.hmap.values()))
        with abbreviated_exception():
            plot = self._init_chart(init_element, ranges)

        if plot.legend:
            self._process_legend(plot)

        self.handles['plot'] = plot
        self.handles['glyph_renderers'] = [r for r in plot.renderers
                                           if isinstance(r, GlyphRenderer)]
        if self.dynamic and not self.static:
            self._update_chart(key, element, ranges)
        else:
            properties = self._plot_properties(key, plot, element)
            plot.update(**properties)

        # Update plot, source and glyph
        self.drawn = True

        return plot

    def update_frame(self, key, ranges=None, plot=None, element=None):
        """
        Updates an existing plot with data corresponding
        to the key.
        """
        reused = isinstance(self.hmap, DynamicMap) and (self.overlaid or self.batched)
        if not reused and element is None:
            element = self._get_frame(key)
        elif element is not None:
            self.current_key = key
            self.current_frame = element

        if element is None or (not self.dynamic and self.static):
            return

        max_cycles = len(self.style._options)
        self.style = self.lookup_options(element, 'style').max_cycles(max_cycles)

        self.set_param(**self.lookup_options(element, 'plot').options)
        ranges = self.compute_ranges(self.hmap, key, ranges)
        ranges = match_spec(element, ranges)
        self.current_ranges = ranges

        self._update_chart(key, element, ranges)


    def _update_chart(self, key, element, ranges):
        with abbreviated_exception():
            new_chart = self._init_chart(element, ranges)
        old_chart = self.handles['plot']
        update_plot(old_chart, new_chart)
        properties = self._plot_properties(key, old_chart, element)
        old_chart.update(**properties)


    @property
    def current_handles(self):
        return self.state.select(type=(ColumnDataSource, DataRange1d, Range1d))


class BoxPlot(ChartPlot):
    """
    BoxPlot generates a box and whisker plot from a BoxWhisker
    Element. This allows plotting the median, mean and various
    percentiles.
    """

    style_opts = ['whisker_color', 'marker'] + line_properties

    def _init_chart(self, element, ranges):
        if BokehBoxPlot is None:
            raise SkipRendering('BoxPlot requires bkcharts to be installed, '
                                'and will be replaced with a native implementation.')
        properties = self.style[self.cyclic_index]
        label = element.dimensions('key', True)
        dframe = element.dframe()

        # Fix for displaying datetimes which are not handled by bokeh
        for kd in element.kdims:
            col = dframe[kd.name]
            if col.dtype.kind in ('M',):
                dframe[kd.name] = [kd.pprint_value(v).replace(':', ';')
                                   for v in col]

        if not element.kdims:
            dframe[''] = ''
            label = ['']

        return BokehBoxPlot(dframe, label=label, values=element.vdims[0].name,
                            **properties)


    def _update_chart(self, key, element, ranges):
        super(BoxPlot, self)._update_chart(key, element, ranges)
        vdim = element.vdims[0].name
        start, end = ranges[vdim]
        self.state.y_range.start = start
        self.state.y_range.end = end

from collections import defaultdict
from functools import partial

import param
import numpy as np

from bokeh.models import DataRange1d, FactorRange, HoverTool

from ...core.util import basestring, dimension_sanitizer
from ...operation.stats import univariate_kde
from .chart import AreaPlot
from .element import (CompositeElementPlot, ColorbarPlot, LegendPlot,
                      fill_properties, line_properties)
from .path import PolygonPlot


class DistributionPlot(AreaPlot):
    """
    DistributionPlot visualizes a distribution of values as a KDE.
    """

    bandwidth = param.Number(default=None, doc="""
        The bandwidth of the kernel for the density estimate.""")

    cut = param.Number(default=3, doc="""
        Draw the estimate to cut * bw from the extreme data points.""")

    filled = param.Boolean(default=True, doc="""
        Whether the bivariate contours should be filled.""")


class BivariatePlot(PolygonPlot):
    """
    Bivariate plot visualizes two-dimensional kernel density
    estimates. Additionally, by enabling the joint option, the
    marginals distributions can be plotted alongside each axis (does
    not animate or compose).
    """

    bandwidth = param.Number(default=None, doc="""
        The bandwidth of the kernel for the density estimate.""")

    cut = param.Number(default=3, doc="""
        Draw the estimate to cut * bw from the extreme data points.""")

    filled = param.Boolean(default=False, doc="""
        Whether the bivariate contours should be filled.""")



class BoxWhiskerPlot(CompositeElementPlot, ColorbarPlot, LegendPlot):

    color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    # X-axis is categorical
    _x_range_type = FactorRange

    # Declare that y-range should auto-range if not bounded
    _y_range_type = DataRange1d

    # Map each glyph to a style group
    _style_groups = {'rect': 'whisker', 'segment': 'whisker',
                     'vbar': 'box', 'hbar': 'box', 'circle': 'outlier'}

    style_opts = (['whisker_'+p for p in line_properties] +\
                  ['box_'+p for p in fill_properties+line_properties] +\
                  ['outlier_'+p for p in fill_properties+line_properties] + ['width', 'cmap'])

    _stream_data = False # Plot does not support streaming data

    def get_extents(self, element, ranges):
        """
        Extents are set to '' and None because x-axis is categorical and
        y-axis auto-ranges.
        """
        yrange = element.range(element.vdims[0], data_range=False)
        return ('', yrange[0], '', yrange[1])

    def _get_axis_labels(self, *args, **kwargs):
        """
        Override axis labels to group all key dimensions together.
        """
        element = self.current_frame
        xlabel = ', '.join([kd.pprint_label for kd in element.kdims])
        ylabel = element.vdims[0].pprint_label
        return xlabel, ylabel, None

    def _glyph_properties(self, plot, element, source, ranges, style):
        properties = dict(style, source=source)
        if self.show_legend and not element.kdims:
            properties['legend'] = element.label
        return properties

    def _get_factors(self, element):
        """
        Get factors for categorical axes.
        """
        if not element.kdims:
            xfactors, yfactors =  [element.label], []
        else:
            factors = [tuple(d.pprint_value(v) for d, v in zip(element.kdims, key))
                       for key in element.groupby(element.kdims).data.keys()]
            factors = [f[0] if len(f) == 1 else f for f in factors]
            xfactors, yfactors = factors, []
        return (yfactors, xfactors) if self.invert_axes else (xfactors, yfactors)

    def get_data(self, element, ranges, style):
        if element.kdims:
            groups = element.groupby(element.kdims).data
        else:
            groups = dict([(element.label, element)])
        vdim = dimension_sanitizer(element.vdims[0].name)

        # Define CDS data
        r1_data, r2_data = ({'index': [], 'top': [], 'bottom': []} for i in range(2))
        s1_data, s2_data = ({'x0': [], 'y0': [], 'x1': [], 'y1': []} for i in range(2))
        w1_data, w2_data = ({'index': [], vdim: []} for i in range(2))
        out_data = defaultdict(list, {'index': [], vdim: []})

        # Define glyph-data mapping
        width = style.get('width', 0.7)
        if self.invert_axes:
            vbar_map = {'y': 'index', 'left': 'top', 'right': 'bottom', 'height': width}
            seg_map = {'y0': 'x0', 'y1': 'x1', 'x0': 'y0', 'x1': 'y1'}
            whisk_map = {'y': 'index', 'x': vdim, 'height': 0.2, 'width': 0.001}
            out_map = {'y': 'index', 'x': vdim}
        else:
            vbar_map = {'x': 'index', 'top': 'top', 'bottom': 'bottom', 'width': width}
            seg_map = {'x0': 'x0', 'x1': 'x1', 'y0': 'y0', 'y1': 'y1'}
            whisk_map = {'x': 'index', 'y': vdim, 'width': 0.2, 'height': 0.001}
            out_map = {'x': 'index', 'y': vdim}
        vbar2_map = dict(vbar_map)

        # Get color values
        if self.color_index is not None:
            cdim = element.get_dimension(self.color_index)
            cidx = element.get_dimension_index(self.color_index)
        else:
            cdim, cidx = None, None

        factors = []
        for key, g in groups.items():
            # Compute group label
            if element.kdims:
                label = tuple(d.pprint_value(v) for d, v in zip(element.kdims, key))
                if len(label) == 1:
                    label = label[0]
            else:
                label = key

            # Add color factor
            if cidx is not None and cidx<element.ndims:
                factors.append(cdim.pprint_value(wrap_tuple(key)[cidx]))
            else:
                factors.append(label)

            # Compute statistics
            vals = g.dimension_values(g.vdims[0])
            if len(vals):
                qmin, q1, q2, q3, qmax = (np.percentile(vals, q=q) for q in range(0,125,25))
                iqr = q3 - q1
                upper = min(q3 + 1.5*iqr, vals.max())
                lower = max(q1 - 1.5*iqr, vals.min())
            else:
                qmin, q1, q2, q3, qmax = 0, 0, 0, 0, 0
                lower, upper = 0, 0
            outliers = vals[(vals>upper) | (vals<lower)]
            # Add to CDS data
            for data in [r1_data, r2_data, w1_data, w2_data]:
                data['index'].append(label)
            for data in [s1_data, s2_data]:
                data['x0'].append(label)
                data['x1'].append(label)
            r1_data['top'].append(q2)
            r2_data['top'].append(q1)
            r1_data['bottom'].append(q3)
            r2_data['bottom'].append(q2)
            s1_data['y0'].append(upper)
            s2_data['y0'].append(lower)
            s1_data['y1'].append(q3)
            s2_data['y1'].append(q1)
            w1_data[vdim].append(lower)
            w2_data[vdim].append(upper)
            if len(outliers):
                out_data['index'] += [label]*len(outliers)
                out_data[vdim] += list(outliers)
                if any(isinstance(t, HoverTool) for t in self.state.tools):
                    for kd, k in zip(element.kdims, wrap_tuple(key)):
                        out_data[dimension_sanitizer(kd.name)] += [k]*len(outliers)

        # Define combined data and mappings
        bar_glyph = 'hbar' if self.invert_axes else 'vbar'
        data = {
            bar_glyph+'_1': r1_data, bar_glyph+'_2': r2_data, 'segment_1': s1_data,
            'segment_2': s2_data, 'rect_1': w1_data, 'rect_2': w2_data,
            'circle': out_data
        }
        mapping = {
            bar_glyph+'_1': vbar_map, bar_glyph+'_2': vbar2_map, 'segment_1': seg_map,
            'segment_2': seg_map, 'rect_1': whisk_map, 'rect_2': whisk_map,
            'circle': out_map
        }

        # Cast data to arrays to take advantage of base64 encoding
        for gdata in [r1_data, r2_data, s1_data, s2_data, w1_data, w2_data, out_data]:
            for k, values in gdata.items():
                gdata[k] = np.array(values)

        # Return if not grouped
        if not element.kdims:
            return data, mapping, style

        # Define color dimension and data
        if cidx is None or cidx>=element.ndims:
            cdim = Dimension('index')
        else:
            r1_data[dimension_sanitizer(cdim.name)] = factors
            r2_data[dimension_sanitizer(cdim.name)] = factors
            factors = list(unique_iterator(factors))

        # Get colors and define categorical colormapper
        cname = dimension_sanitizer(cdim.name)
        cmap = style.get('cmap')
        if cmap is None:
            cycle_style = self.lookup_options(element, 'style')
            styles = cycle_style.max_cycles(len(factors))
            colors = [styles[i].get('box_color', styles[i]['box_fill_color'])
                      for i in range(len(factors))]
            colors = [rgb2hex(c) if isinstance(c, tuple) else c for c in colors]
        else:
            colors = None
        mapper = self._get_colormapper(cdim, element, ranges, style, factors, colors)
        vbar_map['fill_color'] = {'field': cname, 'transform': mapper}
        vbar2_map['fill_color'] = {'field': cname, 'transform': mapper}
        if self.show_legend:
            vbar_map['legend'] = cdim.name

        return data, mapping, style

    

class ViolinPlot(BoxWhiskerPlot):

    bandwidth = param.Number(default=None, doc="""
        Allows supplying explicit bandwidth value rather than relying
        on scott or silverman method.""")

    clip = param.Boolean(default=True, doc="""
        Whether to clip the violins at the max and min of the data.""")

    cut = param.Number(default=5, doc="""
        Draw the estimate to cut * bw from the extreme data points.""")

    # Map each glyph to a style group
    _style_groups = {'patch': 'violin', 'segment': 'stats'}

    style_opts = (['violin_'+p for p in fill_properties+line_properties] +
                 ['stats_'+p for p in line_properties]+['stats_color', 'violin_color'])

    _stat_fns = [partial(np.percentile, q=25), np.median, partial(np.percentile, q=75)]

    def _kde_data(self, el, key, **kwargs):
        vdim = el.vdims[0]
        values = el.dimension_values(vdim)
        if self.clip:
            vdim = vdim(range=(np.nanmin(values), np.nanmax(values)))
            el = el.clone(vdims=[vdim])
        kde = univariate_kde(el, dimension=vdim, **kwargs)
        xs, ys = (kde.dimension_values(i) for i in range(2))
        ys = (ys/ys.max())*0.4
        ys = [key+(sign*y,) for sign, vs in ((-1, ys), (1, ys[::-1])) for y in vs]
        kde =  {'x': np.concatenate([xs, xs[::-1]]), 'y': ys}

        values = el.dimension_values(vdim)
        segments = defaultdict(list)
        for stat_fn in self._stat_fns:
            stat = stat_fn(values)
            sidx = np.argmin(np.abs(xs-stat))
            sx, sy = xs[sidx], ys[sidx]
            segments['x'].append(sx)
            segments['y0'].append(key+(-sy[-1],))
            segments['y1'].append(sy)
        return kde, segments

    def get_data(self, element, ranges, style):
        if element.kdims:
            groups = element.groupby(element.kdims).data
        else:
            groups = dict([((element.label,), element)])

        # Define glyph-data mapping
        if self.invert_axes:
            kde_map = {'x': 'x', 'y': 'y'}
            median_map = {'x0': 'x', 'x1': 'x', 'y0': 'y0', 'y1': 'y1'}
        else:
            kde_map = {'x': 'y', 'y': 'x'}
            median_map = {'y0': 'x', 'y1': 'x', 'x0': 'y0', 'x1': 'y1'}

        # Get color values
        if self.color_index is not None:
            cdim = element.get_dimension(self.color_index)
            cidx = element.get_dimension_index(self.color_index)
        else:
            cdim, cidx = None, None

        elstyle = self.lookup_options(element, 'style')
        kwargs = {'bandwidth': self.bandwidth, 'cut': self.cut}

        factors = []
        patch_data, patch_map = {}, {}
        median_data = defaultdict(list)
        for i, (key, g) in enumerate(groups.items()):
            gkey = 'patch_%d'%i
            kde, median = self._kde_data(g, key, **kwargs)
            for k, v in median.items():
                median_data[k] += v
            patch_data[gkey] = kde
            patch_style = {k[7:]: v for k, v in elstyle[i].items()
                           if k.startswith('violin')}
            patch_map[gkey] = dict(kde_map, **patch_style)
        patch_data['segment_1'] = {k: v if isinstance(v[0], tuple) else np.array(v)
                                   for k, v in median_data.items()}
        patch_map['segment_1'] = median_map
        return patch_data, patch_map, style

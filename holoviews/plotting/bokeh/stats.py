from __future__ import absolute_import, division, unicode_literals

from collections import defaultdict
from functools import partial

import param
import numpy as np

from bokeh.models import FactorRange, Circle, VBar, HBar

from .selection import BokehOverlaySelectionDisplay
from ...core.dimension import Dimension, Dimensioned
from ...core.ndmapping import sorted_context
from ...core.util import (basestring, dimension_sanitizer, wrap_tuple,
                          unique_iterator, isfinite)
from ...operation.stats import univariate_kde
from .chart import AreaPlot
from .element import CompositeElementPlot, ColorbarPlot, LegendPlot
from .path import PolygonPlot
from .styles import fill_properties, line_properties
from .util import decode_bytes


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

    selection_display = BokehOverlaySelectionDisplay()


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

    levels = param.ClassSelector(default=10, class_=(list, int), doc="""
        A list of scalar values used to specify the contour levels.""")

    selection_display = BokehOverlaySelectionDisplay(color_prop='cmap', is_cmap=True)


class BoxWhiskerPlot(CompositeElementPlot, ColorbarPlot, LegendPlot):

    show_legend = param.Boolean(default=False, doc="""
        Whether to show legend for the plot.""")

    # Deprecated options

    color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
        Deprecated in favor of color style mapping, e.g. `box_color=dim('color')`""")

    # X-axis is categorical
    _x_range_type = FactorRange

    # Map each glyph to a style group
    _style_groups = {'segment': 'whisker', 'vbar': 'box', 'hbar': 'box', 'circle': 'outlier'}

    style_opts = (['whisker_'+p for p in line_properties] +
                  ['box_'+p for p in fill_properties+line_properties] +
                  ['outlier_'+p for p in fill_properties+line_properties] +
                  ['width', 'box_width', 'cmap', 'box_cmap'])

    _nonvectorized_styles = ['box_width', 'whisker_width', 'width', 'cmap', 'box_cmap']

    _stream_data = False # Plot does not support streaming data

    selection_display = BokehOverlaySelectionDisplay()

    def get_extents(self, element, ranges, range_type='combined'):
        return super(BoxWhiskerPlot, self).get_extents(
            element, ranges, range_type, 'categorical', element.vdims[0]
        )

    def _get_axis_dims(self, element):
        return element.kdims, element.vdims[0]

    def _glyph_properties(self, plot, element, source, ranges, style, group=None):
        properties = dict(style, source=source)
        if self.show_legend and not element.kdims and self.overlaid:
            properties['legend'] = element.label
        return properties

    def _apply_transforms(self, element, data, ranges, style, group=None):
        if element.ndims > 0:
            element = element.aggregate(function=np.mean)
        else:
            agg = element.aggregate(function=np.mean)
            if isinstance(agg, Dimensioned):
                element = agg
            else:
                element = element.clone([(agg,)])
        return super(BoxWhiskerPlot, self)._apply_transforms(element, data, ranges, style, group)

    def _get_factors(self, element, ranges):
        """
        Get factors for categorical axes.
        """
        if not element.kdims:
            xfactors, yfactors = [element.label], []
        else:
            factors = [key for key in element.groupby(element.kdims).data.keys()]
            if element.ndims > 1:
                factors = sorted(factors)
            factors = [tuple(d.pprint_value(k) for d, k in zip(element.kdims, key))
                       for key in factors]
            factors = [f[0] if len(f) == 1 else f for f in factors]
            xfactors, yfactors = factors, []
        return (yfactors, xfactors) if self.invert_axes else (xfactors, yfactors)

    def _postprocess_hover(self, renderer, source):
        if not isinstance(renderer.glyph, (Circle, VBar, HBar)):
            return
        super(BoxWhiskerPlot, self)._postprocess_hover(renderer, source)

    def _box_stats(self, vals):
        vals = vals[isfinite(vals)]

        if len(vals):
            q1, q2, q3 = (np.percentile(vals, q=q)
                          for q in range(25, 100, 25))
            iqr = q3 - q1
            upper = vals[vals <= q3 + 1.5*iqr].max()
            lower = vals[vals >= q1 - 1.5*iqr].min()
        else:
            q1, q2, q3 = 0, 0, 0
            upper, lower = 0, 0
        outliers = vals[(vals > upper) | (vals < lower)]
        return q1, q2, q3, upper, lower, outliers

    def get_data(self, element, ranges, style):
        if element.kdims:
            with sorted_context(False):
                groups = element.groupby(element.kdims).data
        else:
            groups = dict([(element.label, element)])
        vdim = dimension_sanitizer(element.vdims[0].name)

        # Define CDS data
        r1_data, r2_data = ({'index': [], 'top': [], 'bottom': []} for i in range(2))
        s1_data, s2_data = ({'x0': [], 'y0': [], 'x1': [], 'y1': []} for i in range(2))
        w1_data, w2_data = ({'x0': [], 'y0': [], 'x1': [], 'y1': []} for i in range(2))
        out_data = defaultdict(list, {'index': [], vdim: []})

        # Define glyph-data mapping
        width = style.get('box_width', style.get('width', 0.7))
        whisker_width = style.pop('whisker_width', 0.4)/2.
        if 'width' in style:
            self.param.warning("BoxWhisker width option is deprecated "
                               "use 'box_width' instead.")
        if self.invert_axes:
            vbar_map = {'y': 'index', 'left': 'top', 'right': 'bottom', 'height': width}
            seg_map = {'y0': 'x0', 'y1': 'x1', 'x0': 'y0', 'x1': 'y1'}
            out_map = {'y': 'index', 'x': vdim}
        else:
            vbar_map = {'x': 'index', 'top': 'top', 'bottom': 'bottom', 'width': width}
            seg_map = {'x0': 'x0', 'x1': 'x1', 'y0': 'y0', 'y1': 'y1'}
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
            hover = 'hover' in self.handles

            # Add color factor
            if cidx is not None and cidx<element.ndims:
                factors.append(cdim.pprint_value(wrap_tuple(key)[cidx]))
            else:
                factors.append(label)

            # Compute statistics
            vals = g.dimension_values(g.vdims[0])
            q1, q2, q3, upper, lower, outliers = self._box_stats(vals)

            # Add to CDS data
            for data in [r1_data, r2_data]:
                data['index'].append(label)
            for data in [s1_data, s2_data]:
                data['x0'].append(label)
                data['x1'].append(label)
            for data in [w1_data, w2_data]:
                data['x0'].append(wrap_tuple(label)+(-whisker_width,))
                data['x1'].append(wrap_tuple(label)+(whisker_width,))
            r1_data['top'].append(q2)
            r2_data['top'].append(q1)
            r1_data['bottom'].append(q3)
            r2_data['bottom'].append(q2)
            s1_data['y0'].append(upper)
            s2_data['y0'].append(lower)
            s1_data['y1'].append(q3)
            s2_data['y1'].append(q1)
            w1_data['y0'].append(lower)
            w1_data['y1'].append(lower)
            w2_data['y0'].append(upper)
            w2_data['y1'].append(upper)
            if len(outliers):
                out_data['index'] += [label]*len(outliers)
                out_data[vdim] += list(outliers)
                if hover:
                    for kd, k in zip(element.kdims, wrap_tuple(key)):
                        out_data[dimension_sanitizer(kd.name)] += [k]*len(outliers)
            if hover:
                for kd, k in zip(element.kdims, wrap_tuple(key)):
                    kd_name = dimension_sanitizer(kd.name)
                    if kd_name in r1_data:
                        r1_data[kd_name].append(k)
                    else:
                        r1_data[kd_name] = [k]
                    if kd_name in r2_data:
                        r2_data[kd_name].append(k)
                    else:
                        r2_data[kd_name] = [k]
                if vdim in r1_data:
                    r1_data[vdim].append(q2)
                else:
                    r1_data[vdim] = [q2]
                if vdim in r2_data:
                    r2_data[vdim].append(q2)
                else:
                    r2_data[vdim] = [q2]

        # Define combined data and mappings
        bar_glyph = 'hbar' if self.invert_axes else 'vbar'
        data = {
            bar_glyph+'_1': r1_data, bar_glyph+'_2': r2_data, 'segment_1': s1_data,
            'segment_2': s2_data, 'segment_3': w1_data, 'segment_4': w2_data,
            'circle_1': out_data
        }
        mapping = {
            bar_glyph+'_1': vbar_map, bar_glyph+'_2': vbar2_map, 'segment_1': seg_map,
            'segment_2': seg_map, 'segment_3': seg_map, 'segment_4': seg_map,
            'circle_1': out_map
        }

        # Cast data to arrays to take advantage of base64 encoding
        for gdata in [r1_data, r2_data, s1_data, s2_data, out_data]:
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

        if self.show_legend:
            vbar_map['legend'] = cdim.name

        return data, mapping, style



class ViolinPlot(BoxWhiskerPlot):

    bandwidth = param.Number(default=None, doc="""
        Allows supplying explicit bandwidth value rather than relying
        on scott or silverman method.""")

    clip = param.NumericTuple(default=None, length=2, doc="""
        A tuple of a lower and upper bound to clip the violin at.""")

    cut = param.Number(default=5, doc="""
        Draw the estimate to cut * bw from the extreme data points.""")

    inner = param.ObjectSelector(objects=['box', 'quartiles', 'stick', None],
                                 default='box', doc="""
        Inner visual indicator for distribution values:

          * box - A small box plot
          * stick - Lines indicating each sample value
          * quartiles - Indicates first, second and third quartiles
        """)

    violin_width = param.Number(default=0.8, doc="""
       Relative width of the violin""")

    # Deprecated options

    color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
        Deprecated in favor of color style mapping, e.g. `violin_color=dim('color')`""")

    # Map each glyph to a style group
    _style_groups = {'patches': 'violin', 'segment': 'stats', 'vbar': 'box',
                     'scatter': 'median', 'hbar': 'box'}

    _draw_order = ['patches', 'segment', 'vbar', 'hbar', 'circle', 'scatter']

    style_opts = ([glyph+p for p in fill_properties+line_properties
                   for glyph in ('violin_', 'box_')] +
                  ['stats_'+p for p in line_properties] +
                  ['_'.join([glyph, p]) for p in ('color', 'alpha')
                   for glyph in ('box', 'violin', 'stats', 'median')] +
                  ['cmap', 'box_cmap', 'violin_cmap'])

    _stat_fns = [partial(np.percentile, q=q) for q in [25, 50, 75]]

    selection_display = BokehOverlaySelectionDisplay(color_prop='violin_fill_color')

    def _kde_data(self, el, key, **kwargs):
        vdim = el.vdims[0]
        values = el.dimension_values(vdim)
        if self.clip:
            vdim = vdim(range=self.clip)
            el = el.clone(vdims=[vdim])
        kde = univariate_kde(el, dimension=vdim.name, **kwargs)
        xs, ys = (kde.dimension_values(i) for i in range(2))
        mask = isfinite(ys) & (ys>0) # Mask out non-finite and zero values
        xs, ys = xs[mask], ys[mask]
        ys = (ys/ys.max())*(self.violin_width/2.) if len(ys) else []
        ys = [key+(sign*y,) for sign, vs in ((-1, ys), (1, ys[::-1])) for y in vs]
        xs = np.concatenate([xs, xs[::-1]])
        kde =  {'ys': xs, 'xs': ys}

        bars, segments, scatter = defaultdict(list), defaultdict(list), {}
        values = el.dimension_values(vdim)
        values = values[isfinite(values)]
        if not len(values):
            pass
        elif self.inner == 'quartiles':
            if len(xs):
                for stat_fn in self._stat_fns:
                    stat = stat_fn(values)
                    sidx = np.argmin(np.abs(xs-stat))
                    sx, sy = xs[sidx], ys[sidx]
                    segments['x'].append(sx)
                    segments['y0'].append(key+(-sy[-1],))
                    segments['y1'].append(sy)
        elif self.inner == 'stick':
            if len(xs):
                for value in values:
                    sidx = np.argmin(np.abs(xs-value))
                    sx, sy = xs[sidx], ys[sidx]
                    segments['x'].append(sx)
                    segments['y0'].append(key+(-sy[-1],))
                    segments['y1'].append(sy)
        elif self.inner == 'box':
            xpos = key+(0,)
            q1, q2, q3, upper, lower, _ = self._box_stats(values)
            segments['x'].append(xpos)
            segments['y0'].append(lower)
            segments['y1'].append(upper)
            bars['x'].append(xpos)
            bars['bottom'].append(q1)
            bars['top'].append(q3)
            scatter['x'] = xpos
            scatter['y'] = q2
        return kde, segments, bars, scatter


    def get_data(self, element, ranges, style):
        if element.kdims:
            with sorted_context(False):
                groups = element.groupby(element.kdims).data
        else:
            groups = dict([((element.label,), element)])

        # Define glyph-data mapping
        if self.invert_axes:
            bar_map = {'y': 'x', 'left': 'bottom',
                       'right': 'top', 'height': 0.1}
            kde_map = {'xs': 'ys', 'ys': 'xs'}
            if self.inner == 'box':
                seg_map = {'x0': 'y0', 'x1': 'y1', 'y0': 'x', 'y1': 'x'}
            else:
                seg_map = {'x0': 'x', 'x1': 'x', 'y0': 'y0', 'y1': 'y1'}
            scatter_map = {'x': 'y', 'y': 'x'}
            bar_glyph = 'hbar'
        else:
            bar_map = {'x': 'x', 'bottom': 'bottom',
                       'top': 'top', 'width': 0.1}
            kde_map = {'xs': 'xs', 'ys': 'ys'}
            if self.inner == 'box':
                seg_map = {'x0': 'x', 'x1': 'x', 'y0': 'y0', 'y1': 'y1'}
            else:
                seg_map = {'y0': 'x', 'y1': 'x', 'x0': 'y0', 'x1': 'y1'}
            scatter_map = {'x': 'x', 'y': 'y'}
            bar_glyph = 'vbar'

        kwargs = {'bandwidth': self.bandwidth, 'cut': self.cut}
        mapping, data = {}, {}
        patches_data, seg_data, bar_data, scatter_data = (defaultdict(list) for i in range(4))
        for i, (key, g) in enumerate(groups.items()):
            key = decode_bytes(key)
            kde, segs, bars, scatter = self._kde_data(g, key, **kwargs)
            for k, v in segs.items():
                seg_data[k] += v
            for k, v in bars.items():
                bar_data[k] += v
            for k, v in scatter.items():
                scatter_data[k].append(v)
            for k, v in kde.items():
                patches_data[k].append(v)

        data['patches_1'] = patches_data
        mapping['patches_1'] = kde_map
        if seg_data:
            data['segment_1'] = {k: v if isinstance(v[0], tuple) else np.array(v)
                                 for k, v in seg_data.items()}
            mapping['segment_1'] = seg_map
        if bar_data:
            data[bar_glyph+'_1'] = {k: v if isinstance(v[0], tuple) else np.array(v)
                                    for k, v in bar_data.items()}
            mapping[bar_glyph+'_1'] = bar_map
        if scatter_data:
            data['scatter_1'] = {k: v if isinstance(v[0], tuple) else np.array(v)
                              for k, v in scatter_data.items()}
            mapping['scatter_1'] = scatter_map
        return data, mapping, style

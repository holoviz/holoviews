import param
import numpy as np

from .chart import AreaPlot, ChartPlot
from .path import PolygonPlot
from .plot import AdjoinedPlot


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



class BoxPlot(ChartPlot):
    """
    BoxPlot plots the ErrorBar Element type and supporting
    both horizontal and vertical error bars via the 'horizontal'
    plot option.
    """

    style_opts = ['notch', 'sym', 'whis', 'bootstrap',
                  'conf_intervals', 'widths', 'showmeans',
                  'show_caps', 'showfliers', 'boxprops',
                  'whiskerprops', 'capprops', 'flierprops',
                  'medianprops', 'meanprops', 'meanline']

    _plot_methods = dict(single='boxplot')

    def get_extents(self, element, ranges):
        return (np.NaN,)*4


    def get_data(self, element, ranges, style):
        groups = element.groupby(element.kdims)

        data, labels = [], []

        groups = groups.data.items() if element.kdims else [(element.label, element)]
        for key, group in groups:
            if element.kdims:
                label = ','.join([d.pprint_value(v) for d, v in zip(element.kdims, key)])
            else:
                label = key
            data.append(group[group.vdims[0]])
            labels.append(label)
        style['labels'] = labels
        style = {k: v for k, v in style.items()
                 if k not in ['zorder', 'label']}
        style['vert'] = not self.invert_axes
        format_kdims = [kd(value_format=None) for kd in element.kdims]
        return (data,), style, {'dimensions': [format_kdims, element.vdims[0]]}


    def teardown_handles(self):
        for group in self.handles['artist'].values():
            for v in group:
                v.remove()



class SideBoxPlot(AdjoinedPlot, BoxPlot):

    bgcolor = param.Parameter(default=(1, 1, 1, 0), doc="""
        Make plot background invisible.""")

    border_size = param.Number(default=0, doc="""
        The size of the border expressed as a fraction of the main plot.""")

    xaxis = param.ObjectSelector(default='bare',
                                 objects=['top', 'bottom', 'bare', 'top-bare',
                                          'bottom-bare', None], doc="""
        Whether and where to display the xaxis, bare options allow suppressing
        all axis labels including ticks and xlabel. Valid options are 'top',
        'bottom', 'bare', 'top-bare' and 'bottom-bare'.""")

    yaxis = param.ObjectSelector(default='bare',
                                 objects=['left', 'right', 'bare', 'left-bare',
                                          'right-bare', None], doc="""
        Whether and where to display the yaxis, bare options allow suppressing
        all axis labels including ticks and ylabel. Valid options are 'left',
        'right', 'bare' 'left-bare' and 'right-bare'.""")

    def __init__(self, *args, **kwargs):
        super(SideBoxPlot, self).__init__(*args, **kwargs)
        if self.adjoined:
            self.invert_axes = not self.invert_axes


class ViolinPlot(BoxPlot):
    """
    BoxPlot plots the ErrorBar Element type and supporting
    both horizontal and vertical error bars via the 'horizontal'
    plot option.
    """

    _plot_methods = dict(single='violinplot')

    style_opts = ['showmedians', 'showmeans', 'facecolors',
                  'showextrema', 'bw_method', 'widths',
                  'stats_color', 'alpha', 'edgecolors']

    def init_artists(self, ax, plot_args, plot_kwargs):
        stats_color = plot_kwargs.pop('stats_color', 'black')
        facecolors = plot_kwargs.pop('facecolors', [])
        edgecolors = plot_kwargs.pop('edgecolors', 'black')
        alpha = plot_kwargs.pop('alpha', 1.)
        artists = super(ViolinPlot, self).init_artists(ax, plot_args, plot_kwargs)
        artist = artists['artist']
        for body, color in zip(artist['bodies'], facecolors):
            body.set_facecolors(color)
            body.set_edgecolors(edgecolors)
            body.set_alpha(alpha)
        for stat in ['cmedians', 'cmeans', 'cmaxes', 'cmins', 'cbars']:
            if stat in artist:
                artist[stat].set_edgecolors(stats_color)
        return artists

    def get_data(self, element, ranges, style):
        groups = element.groupby(element.kdims)

        data, labels, colors = [], [], []
        elstyle = self.lookup_options(element, 'style')

        groups = groups.data.items() if element.kdims else [(element.label, element)]
        for i, (key, group) in enumerate(groups):
            if element.kdims:
                label = ','.join([d.pprint_value(v) for d, v in zip(element.kdims, key)])
            else:
                label = key
            data.append(group[group.vdims[0]])
            labels.append(label)
            colors.append(elstyle[i].get('facecolors', 'blue'))
        style['positions'] = range(len(data))
        style['facecolors'] = colors
        style = {k: v for k, v in style.items()
                 if k not in ['zorder', 'label']}
        style['vert'] = not self.invert_axes
        format_kdims = [kd(value_format=None) for kd in element.kdims]
        ticks = {'yticks' if self.invert_axes else 'xticks': list(enumerate(labels))}
        return (data,), style, {'dimensions': [format_kdims, element.vdims[0]], **ticks}

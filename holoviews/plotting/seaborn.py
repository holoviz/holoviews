from __future__ import absolute_import

import matplotlib.pyplot as plt

try:
    import seaborn.apionly as sns
except:
    sns = None

import param

from ..interface.pandas import DFrame, DataFrameView
from ..interface.seaborn import Regression, TimeSeries, Bivariate, Distribution
from ..interface.seaborn import DFrame as SNSFrame
from ..core.options import Store
from .element import ElementPlot
from .pandas import DFrameViewPlot


class FullRedrawPlot(ElementPlot):
    """
    FullRedrawPlot provides an abstract baseclass, defining an
    update_frame method, which completely wipes the axis and
    redraws the plot.
    """

    apply_databounds = param.Boolean(default=False, doc="""
        Enables computing the plot bounds from the data itself.
        Disabled by default since data is often preprocessed,
        before display, changing the bounds.""")

    aspect = param.Parameter(default='square', doc="""
        Aspect ratio defaults to square, 'equal' or numeric values
        are also supported.""")

    rescale_individually = param.Boolean(default=False, doc="""
        Whether to use redraw the axes per map or per element.""")

    show_grid = param.Boolean(default=True, doc="""
        Enables the axis grid.""")

    _abstract = True

    def update_handles(self, axis, view, key, ranges=None):
        if self.zorder == 0 and axis:
            axis.cla()
        self._update_plot(axis, view)



class RegressionPlot(FullRedrawPlot):
    """
    RegressionPlot visualizes Regression Views using the Seaborn
    regplot interface, allowing the user to perform and plot
    linear regressions on a set of scatter points. Parameters
    to the replot function can be supplied via the opts magic.
    """

    style_opts = ['x_estimator', 'x_bins', 'x_ci', 'scatter',
                  'fit_reg', 'color', 'n_boot', 'order',
                  'logistic', 'lowess', 'robust', 'truncate',
                  'scatter_kws', 'line_kws', 'ci', 'dropna',
                  'x_jitter', 'y_jitter', 'x_partial', 'y_partial']

    def __call__(self, ranges=None):
        self._update_plot(self.handles['axis'], self.map.last)
        return self._finalize_axis(self.keys[-1])


    def _update_plot(self, axis, view):
        label = view.label if self.overlaid == 1 else ''
        sns.regplot(view.data[:, 0], view.data[:, 1],
                    ax=axis, label=label,
                    **self.style[self.cyclic_index])



class BivariatePlot(FullRedrawPlot):
    """
    Bivariate plot visualizes two-dimensional kernel density
    estimates using the Seaborn kdeplot function. Additionally,
    by enabling the joint option, the marginals distributions
    can be plotted alongside each axis (does not animate or
    compose).
    """

    rescale_individually = param.Boolean(default=True)

    joint = param.Boolean(default=False, doc="""
        Whether to visualize the kernel density estimate with marginal
        distributions along each axis. Does not animate or compose
        when enabled.""")

    style_opts = ['color', 'alpha', 'err_style', 'interpolate',
                  'ci', 'kind', 'bw', 'kernel', 'cumulative',
                  'shade', 'vertical', 'cmap']

    def __call__(self, ranges=None):
        kdeview = self.map.last
        axis = self.handles['axis']
        self.style = self.style[self.cyclic_index]
        if self.joint and self.subplot:
            raise Exception("Joint plots can't be animated or laid out in a grid.")
        self._update_plot(axis, kdeview)

        return self._finalize_axis(self.keys[-1])


    def _update_plot(self, axis, view):
        if self.joint:
            self.style.pop('cmap', None)
            self.handles['fig'] = sns.jointplot(view.data[:,0],
                                                view.data[:,1],
                                                **self.style).fig
        else:
            label = view.label if self.overlaid == 1 else ''
            sns.kdeplot(view.data, ax=axis, label=label,
                        zorder=self.zorder, **self.style)



class TimeSeriesPlot(FullRedrawPlot):
    """
    TimeSeries visualizes sets of curves using the Seaborn
    tsplot function. This provides functionality to plot
    error bars with various styles alongside the averaged
    curve.
    """

    rescale_individually = param.Boolean(default=False)

    show_frame = param.Boolean(default=False, doc="""
       Disabled by default for clarity.""")

    show_legend = param.Boolean(default=True, doc="""
      Whether to show legend for the plot.""")

    style_opts = ['color', 'alpha', 'err_style', 'interpolate',
                  'ci', 'n_boot', 'err_kws', 'err_palette',
                  'estimator', 'kwargs']

    def __call__(self, ranges=None):
        curveview = self.map.last
        axis = self.handles['axis']
        self.style = self.style[self.cyclic_index]
        self._update_plot(axis, curveview)

        return self._finalize_axis(self.keys[-1])


    def _update_plot(self, axis, view):
        sns.tsplot(view.data, view.xdata, ax=axis, condition=view.label,
                   zorder=self.zorder, **self.style)



class DistributionPlot(FullRedrawPlot):
    """
    DistributionPlot visualizes Distribution Views using the
    Seaborn distplot function. This allows visualizing a 1D
    array as a histogram, kernel density estimate, or rugplot.
    """

    rescale_individually = param.Boolean(default=False)

    show_frame = param.Boolean(default=False, doc="""
       Disabled by default for clarity.""")

    style_opts = ['bins', 'hist', 'kde', 'rug', 'fit',
                  'hist_kws', 'kde_kws', 'rug_kws',
                  'fit_kws', 'color']

    def __call__(self, ranges=None):
        distview = self.map.last
        axis = self.handles['axis']
        self.style = self.style[self.cyclic_index]
        self._update_plot(axis, distview)

        return self._finalize_axis(self.keys[-1])


    def _update_plot(self, axis, view):
        label = view.label if self.overlaid == 1 else ''
        sns.distplot(view.data, ax=axis, label=label, **self.style)



class SNSFramePlot(DFrameViewPlot):
    """
    SNSFramePlot takes an SNSFrame as input and plots the
    contained data using the set plot_type. This largely mirrors
    the way DFramePlot works, however, since most Seaborn plot
    types plot one dimension against another it uses the x and y
    parameters, which can be set on the SNSFrame.
    """

    plot_type = param.ObjectSelector(default='scatter_matrix',
                                     objects=['interact', 'regplot',
                                              'lmplot', 'corrplot',
                                              'plot', 'boxplot',
                                              'hist', 'scatter_matrix',
                                              'autocorrelation_plot',
                                              'pairgrid', 'facetgrid',
                                              'pairplot', 'violinplot',
                                              'factorplot'
                                          ],
                                     doc="""
        Selects which Seaborn plot type to use, when visualizing the
        SNSFrame. The options that can be passed to the plot_type are
        defined in dframe_options.""")

    dframe_options = dict(DFrameViewPlot.dframe_options,
                          **{'regplot':   RegressionPlot.style_opts,
                             'factorplot': ['kind', 'col', 'aspect', 'row',
                                            'col_wrap', 'ci', 'linestyles',
                                            'markers', 'palette', 'dodge',
                                            'join', 'size', 'legend',
                                            'sharex', 'sharey', 'hue', 'estimator'],
                             'boxplot':   [],
                             'violinplot':['groupby', 'positions',
                                           'inner', 'bw', 'cut'],
                             'lmplot':    ['hue', 'col', 'row', 'palette',
                                           'sharex', 'dropna', 'legend'],
                             'corrplot':  ['annot', 'sig_stars', 'sig_tail',
                                           'sig_corr', 'cmap', 'cmap_range',
                                           'cbar'],
                             'interact':  ['filled', 'cmap', 'colorbar',
                                           'levels', 'logistic', 'contour_kws',
                                           'scatter_kws'],
                             'pairgrid':  ['hue', 'hue_order', 'palette',
                                           'hue_kws', 'vars', 'x_vars', 'y_vars'
                                           'size', 'aspect', 'despine', 'map',
                                           'map_diag', 'map_offdiag',
                                           'map_upper', 'map_lower'],
                             'pairplot':  ['hue', 'hue_order', 'palette',
                                           'vars', 'x_vars', 'y_vars', 'diag_kind',
                                           'kind', 'plot_kws', 'diag_kws', 'grid_kws'],
                             'facetgrid': ['hue', 'row', 'col', 'col_wrap',
                                           'map', 'sharex', 'sharey', 'size',
                                           'aspect', 'palette', 'row_order',
                                           'col_order', 'hue_order', 'legend',
                                           'legend_out', 'xlim', 'ylim', 'despine'],
                          })

    style_opts = list({opt for opts in dframe_options.values() for opt in opts})

    def __init__(self, view, **params):
        if self.plot_type in ['pairgrid', 'pairplot', 'facetgrid']:
            self._create_fig = False
        super(SNSFramePlot, self).__init__(view, **params)


    def __call__(self, ranges=None):
        dfview = self.map.last
        axis = self.handles['axis']
        self._validate(dfview)

        self._update_plot(axis, dfview)
        if 'fig' in self.handles and self.handles['fig'] != plt.gcf():
            self.handles['fig'] = plt.gcf()

        return self._finalize_axis(self.keys[-1])


    def _process_style(self, styles):
        styles = super(SNSFramePlot, self)._process_style(styles)
        if self.plot_type not in DFrameViewPlot.params()['plot_type'].objects:
            styles.pop('figsize', None)
        return styles


    def _validate(self, dfview):
        super(SNSFramePlot, self)._validate(dfview)
        multi_dim = dfview.ndims > 1
        if self.subplot and multi_dim and self.plot_type == 'lmplot':
            raise Exception("Multiple %s plots cannot be composed."
                            % self.plot_type)

    def update_frame(self, key, ranges=None):
        view = self.map.get(key, None)
        axis = self.handles['axis']
        if axis:
            axis.set_visible(view is not None)
        axis_kwargs = self.update_handles(axis, view, key, ranges)
        if axis:
            self._finalize_axis(key, **(axis_kwargs if axis_kwargs else {}))


    def _update_plot(self, axis, view):
        style = self._process_style(self.style[self.cyclic_index])
        if self.plot_type == 'factorplot':
            opts = dict(style, **({'hue': view.x2} if view.x2 else {}))
            sns.factorplot(x=view.x, y=view.y, data=view.data, **opts)
        elif self.plot_type == 'regplot':
            sns.regplot(x=view.x, y=view.y, data=view.data,
                        ax=axis, **style)
        elif self.plot_type == 'boxplot':
            style.pop('return_type', None)
            style.pop('figsize', None)
            sns.boxplot(view.data[view.y], view.data[view.x], ax=axis,
                        **style)
        elif self.plot_type == 'violinplot':
            sns.violinplot(view.data[view.y], view.data[view.x], ax=axis,
                           **style)
        elif self.plot_type == 'interact':
            sns.interactplot(view.x, view.x2, view.y,
                             data=view.data, ax=axis, **style)
        elif self.plot_type == 'corrplot':
            sns.corrplot(view.data, ax=axis, **style)
        elif self.plot_type == 'lmplot':
            sns.lmplot(x=view.x, y=view.y, data=view.data,
                       ax=axis, **style)
        elif self.plot_type in ['pairplot', 'pairgrid', 'facetgrid']:
            style_keys = list(style.keys())
            map_opts = [(k, style.pop(k)) for k in style_keys if 'map' in k]
            if self.plot_type == 'pairplot':
                g = sns.pairplot(view.data, **style)
            elif self.plot_type == 'pairgrid':
                g = sns.PairGrid(view.data, **style)
            elif self.plot_type == 'facetgrid':
                g = sns.FacetGrid(view.data, **style)
            for opt, args in map_opts:
                plot_fn = getattr(sns, args[0]) if hasattr(sns, args[0]) else getattr(plt, args[0])
                getattr(g, opt)(plot_fn, *args[1:])
            plt.close(self.handles['fig'])
            self.handles['fig'] = plt.gcf()
        else:
            super(SNSFramePlot, self)._update_plot(axis, view)


Store.registry.update({TimeSeries: TimeSeriesPlot,
                       Bivariate: BivariatePlot,
                       Distribution: DistributionPlot,
                       Regression: RegressionPlot,
                       SNSFrame: SNSFramePlot,
                       DFrame: SNSFramePlot,
                       DataFrameView: SNSFramePlot})

from __future__ import absolute_import

from matplotlib import pyplot as plt

import param

from .. import View
from ..interface.pandas import DFrame, DFrameStack, DFrameOverlay, DataFrameView, pd
from .viewplots import Plot



class DFramePlot(Plot):
    """
    A high-level plot, which will plot any DataFrameView or DFrameStack type
    including DataOverlays.

    A generic plot that visualizes DFrameStacks containing DFrameOverlay or
    DataFrameView objects.
    """

    _stack_type = DFrameStack
    _view_type = DFrameOverlay

    style_opts = param.List(default=[], constant=True, doc="""
     DataPlot renders overlay layers which individually have style
     options but DataPlot itself does not.""")

    def __call__(self, axis=None, lbrt=None, **kwargs):

        ax = self._axis(axis, None)

        stacks = self._stack.split_overlays()

        for zorder, stack in enumerate(stacks):
            plotopts = View.options.plotting(stack).opts

            plotype = Plot.defaults[stack.type]
            plot = plotype(stack, size=self.size,
                           show_xaxis=self.show_xaxis, show_yaxis=self.show_yaxis,
                           show_legend=self.show_legend, show_title=self.show_title,
                           show_grid=self.show_grid, zorder=zorder,
                           **dict(plotopts, **kwargs))
            plot.aspect = self.aspect

            plot(ax)
            self.plots.append(plot)

        if axis is None: plt.close(self.handles['fig'])
        return ax if axis else self.handles['fig']


    def update_frame(self, n, lbrt=None):
        n = n if n < len(self) else len(self) - 1
        for zorder, plot in enumerate(self.plots):
            plot.update_frame(n)



class DFrameViewPlot(Plot):
    """
    DFramePlot provides a wrapper around Pandas dataframe plots.
    It takes a single DataFrameView or DFrameStack as input and plots it using
    the plotting command selected via the plot_type.

    The plot_options specifies the valid options to be supplied
    to the selected plot_type via options.style_opts.
    """

    plot_type = param.ObjectSelector(default='boxplot', objects=['plot', 'boxplot',
                                                                 'hist', 'scatter_matrix',
                                                                 'autocorrelation_plot'],
                                     doc="""Selects which Pandas plot type to use.""")

    dframe_options = {'plot': ['kind', 'stacked', 'xerr',
                               'yerr', 'share_x', 'share_y',
                               'table', 'style', 'x', 'y',
                               'secondary_y', 'legend',
                               'logx', 'logy', 'position',
                               'colormap', 'mark_right'],
                      'hist': ['column', 'by', 'grid',
                               'xlabelsize', 'xrot',
                               'ylabelsize', 'yrot',
                               'sharex', 'sharey', 'hist',
                               'layout', 'bins'],
                      'boxplot': ['column', 'by', 'fontsize',
                                  'layout', 'grid', 'rot'],
                      'scatter_matrix': ['alpha', 'grid', 'diagonal',
                                         'marker', 'range_padding',
                                         'hist_kwds', 'density_kwds'],
                      'autocorrelation': ['kwds']}


    class classproperty(object):
        """
        Adds a getter property to a class.
        """
        def __init__(self, f):
            self.f = f
        def __get__(self, obj, owner):
            return self.f(owner)


    @classproperty
    def style_opts(cls):
        """
        Concatenates the Pandas plot_options to make all the options
        available via the StyleOpts interface.
        """
        opt_set = set()
        for opts in cls.dframe_options.values():
            opt_set |= set(opts)
        return list(opt_set)

    _stack_type = DFrameStack
    _view_type = DataFrameView


    def __call__(self, axis=None, cyclic_index=0, lbrt=None):
        dfview = self._stack.last
        composed = axis is not None

        if composed and self.plot_type == 'scatter_matrix':
            raise Exception("Scatter Matrix plots can't be composed.")
        elif composed and len(dfview.dimensions) > 1 and self.plot_type in ['hist']:
            raise Exception("Multiple %s plots cannot be composed." % self.plot_type)

        title = None if self.zorder > 0 else self._format_title(-1)
        self.ax = self._axis(axis, title)

        # Process styles
        self.style = View.options.style(dfview)[cyclic_index]
        styles = self.style.keys()
        for k in styles:
            if k not in self.dframe_options[self.plot_type]:
                self.warning('Plot option %s does not apply to %s plot type.' % (k, self.plot_type))
                self.style.pop(k)
        if self.plot_type not in ['autocorrelation_plot']:
            self.style['figsize'] = self.size

        # Legacy fix for Pandas, can be removed for Pandas >0.14
        if self.plot_type == 'boxplot':
            self.style['return_type'] = 'axes'

        self._update_plot(dfview)

        if not axis:
            fig = self.handles.get('fig', plt.gcf())
            plt.close(fig)
        return self.ax if axis else self.handles.get('fig', plt.gcf())


    def _update_plot(self, dfview):
        if self.plot_type == 'scatter_matrix':
            pd.scatter_matrix(dfview.data, ax=self.ax, **self.style)
        elif self.plot_type == 'autocorrelation_plot':
            pd.tools.plotting.autocorrelation_plot(dfview.data, ax=self.ax, **self.style)
        else:
            getattr(dfview.data, self.plot_type)(ax=self.ax, **self.style)


    def update_frame(self, n, lbrt=None):
        """
        Update the plot for an animation.
        """
        n = n if n < len(self) else len(self) - 1
        dfview = list(self._stack.values())[n]
        if not self.plot_type in ['hist', 'scatter_matrix']:
            if self.zorder == 0: self.ax.cla()
            self.handles['title'] = self.ax.set_title('')
            self._update_title(n)
        self._update_plot(dfview)
        plt.draw()


Plot.defaults.update({DataFrameView: DFrameViewPlot,
                      DFrame: DFrameViewPlot,
                      DFrameOverlay: DFramePlot})

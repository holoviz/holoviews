import param
import numpy as np

from ..core import Dimension, Dataset, NdOverlay
from ..core.dimension import Dimension
from ..core.operation import Operation
from ..core.util import basestring, find_minmax, cartesian_product
from ..element import Curve, Area, Image

from .element import contours


class univariate_kde(Operation):

    bw_method = param.ObjectSelector(default='scott', objects=['scott', 'silverman'], doc="""
        Method of automatically determining KDE bandwidth""")

    bandwidth = param.Number(default=None, doc="""
        Allows supplying explicit bandwidth value rather than relying on scott or silverman method.""")

    bin_range = param.NumericTuple(default=None, length=2,  doc="""
        Specifies the range within which to compute the KDE.""")

    dimension = param.String(default=None, doc="""
        Along which dimension of the Element to compute the KDE.""")

    filled = param.Boolean(default=False, doc="""
        Controls whether to return filled or unfilled KDE.""")

    n_samples = param.Integer(default=100, doc="""
        Number of samples to compute the KDE over.""")
    
    groupby = param.ClassSelector(default=None, class_=(basestring, Dimension), doc="""
      Defines a dimension to group the Histogram returning an NdOverlay of Histograms.""")

    def _process(self, element, key=None):
        if self.p.groupby:
            if not isinstance(element, Dataset):
                raise ValueError('Cannot use histogram groupby on non-Dataset Element')
            grouped = element.groupby(self.p.groupby, group_type=Dataset, container_type=NdOverlay)
            self.p.groupby = None
            return grouped.map(self._process, Dataset)

        try:
            from scipy import stats
        except ImportError:
            raise ImportError('%s operation requires SciPy to be installed.' % type(self).__name__)

        if self.p.dimension:
            selected_dim = self.p.dimension
        else:
            selected_dim = [d.name for d in element.vdims + element.kdims][0]
        data = element.dimension_values(selected_dim)
        bin_range = find_minmax((np.nanmin(data), np.nanmax(data)), (0, -float('inf')))\
            if self.p.bin_range is None else self.p.bin_range

        xs = np.linspace(bin_range[0], bin_range[1], self.p.n_samples)
        kde = stats.gaussian_kde(data)
        if self.p.bandwidth:
            kde.set_bandwidth(self.p.bandwidth)
        ys = kde.evaluate(xs)

        vdims = [Dimension('{}_density'.format(selected_dim), 
                           label='{} Density'.format(selected_dim))]

        element_type = Area if self.p.filled else Curve
        return Area((xs, ys), kdims=[selected_dim], vdims=vdims)


    
class bivariate_kde(Operation):

    contours = param.Boolean(default=False)

    bw_method = param.ObjectSelector(default='scott', objects=['scott', 'silverman'], doc="""
        Method of automatically determining KDE bandwidth""")

    bandwidth = param.Number(default=None, doc="""
        Allows supplying explicit bandwidth value rather than relying on scott or silverman method.""")

    bin_range = param.NumericTuple(default=None, length=2,  doc="""
        Specifies the range within which to compute the KDE.""")

    filled = param.Boolean(default=False, doc="""
        Controls whether to return filled or unfilled contours.""")

    n_samples = param.Integer(default=100, doc="""
        Number of samples to compute the KDE over.""")

    x_range  = param.NumericTuple(default=None, length=2, doc="""
       The x_range as a tuple of min and max x-value. Auto-ranges
       if set to None.""")

    y_range  = param.NumericTuple(default=None, length=2, doc="""
       The x_range as a tuple of min and max y-value. Auto-ranges
       if set to None.""")

    def _process(self, element, key=None):
        try:
            from scipy import stats
        except ImportError:
            raise ImportError('%s operation requires SciPy to be installed.' % type(self).__name__)
                           
        data = element.array([0, 1]).T
        bin_range = find_minmax((np.nanmin(data), np.nanmax(data)), (0, -float('inf')))\
            if self.p.bin_range is None else self.p.bin_range

        xmin, xmax = self.p.x_range or element.range(0)
        ymin, ymax = self.p.y_range or element.range(1)
        kde = stats.gaussian_kde(data)
        if self.p.bandwidth:
            kde.set_bandwidth(self.p.bandwidth)
        xs = np.linspace(xmin, xmax, self.p.n_samples)
        ys = np.linspace(ymin, ymax, self.p.n_samples)
        xx, yy = cartesian_product([xs, ys], False)
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kde(positions).T, xx.shape)

        img = Image((xs, ys, f.T), kdims=element.dimensions()[:2], vdims=['Density'])
        if self.p.contours:
            return contours(img, filled=self.p.filled, )
        return img

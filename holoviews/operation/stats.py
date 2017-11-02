import param
import numpy as np

from ..core import Dimension, Dataset, NdOverlay
from ..core.dimension import Dimension
from ..core.operation import Operation
from ..core.options import Compositor, Store, Options, StoreOptions
from ..core.util import basestring, find_minmax, cartesian_product
from ..element import (Curve, Area, Image, Distribution, Bivariate,
                       Contours, Polygons)

from .element import contours


def _kde_support(bin_range, bw, gridsize, cut, clip):
    """Establish support for a kernel density estimate."""
    kmin, kmax = bin_range[0] - bw * cut, bin_range[1] + bw * cut
    if clip[0] is not None and np.isfinite(clip[0]):
        kmin = max(kmin, clip[0])
    if clip[1] is not None and np.isfinite(clip[1]):
        kmax = max(kmax, clip[1])
    return np.linspace(kmin, kmax, gridsize)


class univariate_kde(Operation):
    """
    Computes a 1D kernel density estimate (KDE) along the supplied
    dimension. Kernel density estimation is a non-parametric way to
    estimate the probability density function of a random variable.

    The KDE works by placing a Gaussian kernel at each sample with
    the supplied bandwidth. These kernels are then summed to produce
    the density estimate. By default a good bandwidth is determined
    using the bw_method but it may be overridden by an explicit value.
    """

    bw_method = param.ObjectSelector(default='scott', objects=['scott', 'silverman'], doc="""
        Method of automatically determining KDE bandwidth""")

    bandwidth = param.Number(default=None, doc="""
        Allows supplying explicit bandwidth value rather than relying on scott or silverman method.""")

    cut = param.Number(default=3, doc="""
        Draw the estimate to cut * bw from the extreme data points.""")

    bin_range = param.NumericTuple(default=None, length=2,  doc="""
        Specifies the range within which to compute the KDE.""")

    dimension = param.String(default=None, doc="""
        Along which dimension of the Element to compute the KDE.""")

    filled = param.Boolean(default=True, doc="""
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

        params = {}
        if isinstance(element, Distribution):
            selected_dim = element.kdims[0]
            if element.group != type(element).__name__:
                params['group'] = element.group
            params['label'] = element.label
            vdim = element.vdims[0]
            vdim_name = '{}_density'.format(selected_dim.name)
            vdim_label = '{} Density'.format(selected_dim.label)
            vdims = [vdim(vdim_name, label=vdim_label) if vdim.name == 'Density' else vdim]
        else:
            if self.p.dimension:
                selected_dim = element.get_dimension(self.p.dimension)
            else:
                selected_dim = [d.name for d in element.vdims + element.kdims][0]
            vdim_name = '{}_density'.format(selected_dim.name)
            vdim_label = '{} Density'.format(selected_dim.label)
            vdims = [Dimension(vdim_nam, label=vdim_label)]

        data = element.dimension_values(selected_dim)
        bin_range = self.p.bin_range or element.range(selected_dim)
        if bin_range == (0, 0) or any(not np.isfinite(r) for r in bin_range):
            bin_range = (0, 1)

        data = data[np.isfinite(data)]
        if len(data):
            kde = stats.gaussian_kde(data)
            if self.p.bandwidth:
                kde.set_bandwidth(self.p.bandwidth)
            bw = kde.scotts_factor() * data.std(ddof=1)
            xs = _kde_support(bin_range, bw, self.p.n_samples, self.p.cut, selected_dim.range)
            ys = kde.evaluate(xs)
        else:
            xs = np.linspace(bin_range[0], bin_range[1], self.p.n_samples)
            ys = np.full_like(xs, 0)

        element_type = Area if self.p.filled else Curve
        return element_type((xs, ys), kdims=[selected_dim], vdims=vdims, **params)



class bivariate_kde(Operation):
    """
    Computes a 2D kernel density estimate (KDE) of the first two
    dimensions in the input data. Kernel density estimation is a
    non-parametric way to estimate the probability density function of
    a random variable.

    The KDE works by placing 2D Gaussian kernel at each sample with
    the supplied bandwidth. These kernels are then summed to produce
    the density estimate. By default a good bandwidth is determined
    using the bw_method but it may be overridden by an explicit value.
    """

    contours = param.Boolean(default=True, doc="""
        Whether to compute contours from the KDE, determines whether to
        return an Image or Contours/Polygons.""")

    bw_method = param.ObjectSelector(default='scott', objects=['scott', 'silverman'], doc="""
        Method of automatically determining KDE bandwidth""")

    bandwidth = param.Number(default=None, doc="""
        Allows supplying explicit bandwidth value rather than relying
        on scott or silverman method.""")

    cut = param.Number(default=3, doc="""
        Draw the estimate to cut * bw from the extreme data points.""")

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

        xdim, ydim = element.dimensions()[:2]
        params = {}
        if isinstance(element, Bivariate):
            if element.group != type(element).__name__:
                params['group'] = element.group
            params['label'] = element.label
            vdim = element.vdims[0]
        else:
            vdim = 'Density'

        data = element.array([0, 1]).T
        xmin, xmax = self.p.x_range or element.range(0)
        ymin, ymax = self.p.y_range or element.range(1)
        if any(not np.isfinite(v) for v in (xmin, xmax)):
            xmin, xmax = -0.5, 0.5
        if any(not np.isfinite(v) for v in (ymin, ymax)):
            ymin, ymax = -0.5, 0.5
        if len(element) > 1:
            kde = stats.gaussian_kde(data)
            if self.p.bandwidth:
                kde.set_bandwidth(self.p.bandwidth)
            bw = kde.scotts_factor() * data.std(ddof=1)
            xs = _kde_support((xmin, xmax), bw, self.p.n_samples, self.p.cut, xdim.range)
            ys = _kde_support((ymin, ymax), bw, self.p.n_samples, self.p.cut, ydim.range)
            xx, yy = cartesian_product([xs, ys], False)
            positions = np.vstack([xx.ravel(), yy.ravel()])
            f = np.reshape(kde(positions).T, xx.shape)
        elif self.p.contours:
            eltype = Polygons if self.p.filled else Contours
            return eltype([], kdims=[xdim, ydim], vdims=[vdim])
        else:
            xs = np.linspace(xmin, xmax, self.p.n_samples)
            ys = np.linspace(ymin, ymax, self.p.n_samples)
            f = np.zeros((self.p.n_samples, self.p.n_samples))

        img = Image((xs, ys, f.T), kdims=element.dimensions()[:2], vdims=[vdim], **params)
        if self.p.contours:
            cntr = contours(img, filled=self.p.filled)
            return cntr.clone(cntr.data[1:], **params)
        return img

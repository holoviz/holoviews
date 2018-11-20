import operator

import numpy as np

from ..core.dimension import Dimension
from ..core.util import basestring, unique_iterator
from ..element import Graph


def norm_fn(values, min=None, max=None):
    """
    min-max normalizes to scale data into 0-1 range.

    Arguments
    ---------
    values: np.ndarray
       Array of values to be binned
    min: float (optional)
       Lower bound of normalization range
    max: float (optional)
       Upper bound of normalization range

    Returns
    -------
    normalized: np.ndarray
       Array of normalized values
    """
    min = np.min(values) if min is None else min
    max = np.max(values) if max is None else max
    return (values - min) / (max-min)


def bin_fn(values, bins, labels=None):
    """
    Bins data into declared bins. By default each bin is labelled
    with bin center values but an explicit list of bin labels may be
    defined.

    Arguments
    ---------
    values: np.ndarray
       Array of values to be binned
    bins: np.ndarray or list
       Bin edges to place bins into
    labels: np.ndarray or list (optional)
       Labels for bins should have length N-1 compared to bins

    Returns
    -------
    binned: np.ndarray
       Array of binned values
    """
    bins = np.asarray(bins)
    if labels is None:
        labels = (bins[:-1] + np.diff(bins)/2.)
    else:
        labels = np.asarray(labels)
    dtype = 'float' if labels.dtype.kind == 'f' else 'O'
    binned = np.full_like(values, (np.nan if dtype == 'f' else None), dtype=dtype)
    for lower, upper, label in zip(bins[:-1], bins[1:], labels):
        condition = (values > lower) & (values <= upper)
        binned[np.where(condition)[0]] = label
    return binned


def cat_fn(values, categories, empty=None):
    """
    Replaces discrete values in input array into a fixed set of
    categories defined either as a list or dictionary.

    Arguments
    ---------
    values: np.ndarray
       Array of values to be categorized
    categories: list or dict
       Categories to assign to input values
    empty: any (optional)
       Value assigned to input values no category could be assigned to

    Returns
    -------
    categorized: np.ndarray
       Array of categorized values
    """
    uniq_cats = list(unique_iterator(values))
    cats = []
    for c in values:
        if isinstance(categories, list):
            cat_ind = uniq_cats.index(c)
            if cat_ind < len(categories):
                cat = categories[cat_ind]
            else:
                cat = empty
        else:
            cat = categories.get(c, empty)
        cats.append(cat)
    return np.asarray(cats)



class dim(object):
    """
    dim transform objects are a way to express deferred transformations
    on HoloViews Datasets. Dims support all mathematical operations
    and NumPy ufuncs, and provide a number of useful methods for normalizing,
    binning and categorizing data.
    """

    _op_registry = {'norm': norm_fn, 'bin': bin_fn, 'cat': cat_fn}

    def __init__(self, obj, fn=None, other=None, reverse=False, **kwargs):
        ops = []
        if isinstance(obj, basestring):
            self.dimension = Dimension(obj)
        elif isinstance(obj, Dimension):
            self.dimension = obj
        else:
            self.dimension = obj.dimension
            ops = obj.ops
        if isinstance(fn, str) or fn in self._op_registry:
            fn = self._op_registry.get(fn)
            if fn is None:
                raise ValueError('dim transform %s not found' % fn)
        if fn is not None:
            ops = ops + [{'other': other, 'fn': fn, 'kwargs': kwargs,
                          'reverse': reverse}]
        self.ops = ops

    @classmethod
    def register(cls, key, function):
        """
        Register a custom dim transform function which can from then
        on be referenced by the key.
        """
        cls._op_registry[key] = function

    # Unary operators
    def __abs__(self): return dim(self, operator.abs)
    def __neg__(self): return dim(self, operator.neg)
    def __pos__(self): return dim(self, operator.pos)

    # Binary operators
    def __add__(self, other):       return dim(self, operator.add, other)
    def __div__(self, other):       return dim(self, operator.div, other)
    def __floordiv__(self, other):  return dim(self, operator.floordiv, other)
    def __mod__(self, other):       return dim(self, operator.mod, other)
    def __mul__(self, other):       return dim(self, operator.mul, other)
    def __pow__(self, other):       return dim(self, operator.pow, other)
    def __sub__(self, other):       return dim(self, operator.sub, other)
    def __truediv__(self, other):   return dim(self, operator.truediv, other)

    # Reverse binary operators
    def __radd__(self, other):      return dim(self, operator.add, other, True)
    def __rdiv__(self, other):      return dim(self, operator.div, other, True)
    def __rfloordiv__(self, other): return dim(self, operator.floordiv, other, True)
    def __rmod__(self, other):      return dim(self, operator.mod, other, True)
    def __rmul__(self, other):      return dim(self, operator.mul, other, True)
    def __rsub__(self, other):      return dim(self, operator.sub, other, True)
    def __rtruediv__(self, other):  return dim(self, operator.truediv, other, True)

    ## NumPy operations
    def __array_ufunc__(self, *args, **kwargs):
        ufunc = getattr(args[0], args[1])
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return dim(self, ufunc, **kwargs)

    def astype(self, dtype):     return dim(self, np.asarray, dtype=dtype)
    def cumsum(self, **kwargs):  return dim(self, np.cumsum, **kwargs)
    def max(self, **kwargs):     return dim(self, np.max, **kwargs)
    def mean(self, **kwargs):    return dim(self, np.mean, **kwargs)
    def min(self, **kwargs):     return dim(self, np.min, **kwargs)
    def round(self, decimals=0): return dim(self, np.round, decimals=decimals)
    def sum(self, **kwargs):     return dim(self, np.sum, **kwargs)
    def std(self, **kwargs):     return dim(self, np.std, **kwargs)
    def var(self, **kwargs):     return dim(self, np.var, **kwargs)

    ## Custom functions

    def bin(self, bins, labels=None):
        """
        Replaces discrete values in input array into a fixed set of
        categories defined either as a list or dictionary.

        Arguments
        ---------
        categories: list or dict
           Categories to assign to input values
        empty: any (optional)
           Value assigned to input values no category could be assigned to
        """
        return dim(self, bin_fn, bins, labels=labels)

    def cat(self, categories, empty=None):
        """
        Bins data into declared bins. By default each bin is labelled
        with bin center values but an explicit list of bin labels may be
        defined.

        Arguments
        ---------
        bins: np.ndarray or list
           Bin edges to place bins into
        labels: np.ndarray or list (optional)
           Labels for bins should have length N-1 compared to bins
        """
        return dim(self, cat_fn, categories=categories, empty=empty)

    def norm(self):
        """
        min-max normalizes to scale data into 0-1 range.
        """
        return dim(self, norm_fn)

    # Other methods
    
    def applies(self, dataset):
        """
        Determines whether the dim transform can be applied to the
        Dataset, i.e. whether all referenced dimensions can be
        resolved.
        """
        if isinstance(self.dimension, dim):
            applies = self.dimension.applies(dataset)
        else:
            applies = dataset.get_dimension(self.dimension) is not None
            if isinstance(dataset, Graph) and not applies:
                applies = dataset.nodes.get_dimension(self.dimension) is not None
        for op in self.ops:
            other = op.get('other')
            if other is None:
                continue
            elif isinstance(other, basestring):
                applies &= dim(other).applies(dataset)
            elif isinstance(other, dim):
                applies &= other.applies(dataset)
        return applies

    def apply(self, dataset, flat=False, expanded=None, ranges={}):
        """
        Evaluates the transform on the supplied dataset.
        """
        if expanded is None:
            expanded = not ((dataset.interface.gridded and self.dimension in dataset.kdims) or
                            (dataset.interface.multi and dataset.interface.isscalar(dataset, self.dimension)))
        if isinstance(dataset, Graph):
            dataset = dataset if self.dimension in dataset else dataset.nodes
        data = dataset.dimension_values(self.dimension, expanded=expanded, flat=flat)
        for o in self.ops:
            other = o['other']
            if other is not None:
                if isinstance(other, dim):
                    other = other.apply(dataset, flat, expanded, ranges)
                args = (other, data) if o['reverse'] else (data, other)
            else:
                args = (data,)
            drange = ranges.get(str(self), {})
            drange = drange.get('combined', drange)
            if o['fn'] == norm_fn and drange != {}:
                data = o['fn'](data, *drange)
            else:
                data = o['fn'](*args, **o['kwargs'])
        return data

    def __repr__(self):
        op_repr = "'%s'" % self.dimension
        for o in self.ops:
            arg = ', %r' % o['other'] if o['other'] else ''
            kwargs = sorted(o['kwargs'].items(), key=operator.itemgetter(0))
            kwargs = ', %s' % ', '.join(['%s=%s' % item for item in kwargs]) if kwargs else ''
            op_repr = '{fn}({repr}{arg}{kwargs})'.format(fn=o['fn'].__name__, repr=op_repr,
                                                         arg=arg, kwargs=kwargs)
        return op_repr


import operator
from itertools import zip_longest

import numpy as np

from ..core.dimension import Dimension
from ..core.util import basestring, unique_iterator, isfinite
from ..element import Graph


def norm_fn(values, min=None, max=None):
    min = np.min(values) if min is None else min
    max = np.max(values) if max is None else max
    return (values - min) / (max-min)


def bin_fn(values, bins, labels=None):
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


def str_fn(values):
    return np.asarray([str(v) for v in values])


def int_fn(values):
    return values.astype(int)


class dim(object):

    _op_registry = {'norm': norm_fn, 'bin': bin_fn, 'cat': cat_fn,
                    str: str_fn, int: int_fn}

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
                raise ValueError('Operation function %s not found' % fn)
        if fn is not None:
            ops = ops + [{'other': other, 'fn': fn, 'kwargs': kwargs,
                          'reverse': reverse}]
        self.ops = ops

    @classmethod
    def resolve_spec(cls, op_spec):
        """
        Converts an op spec, i.e. a string or tuple declaring an op
        or a nested spec of ops into an op instance.
        """
        if isinstance(op_spec, basestring):
            return cls(op_spec)
        elif isinstance(op_spec, tuple):
            combined = zip_longest(op_spec, (None, None, None, {}))
            obj, fn, other, kwargs = (o2 if o1 is None else o1 for o1, o2 in combined)
            if isinstance(obj, tuple):
                obj = cls.resolve_spec(obj)
            return cls(obj, fn, other, **kwargs)
        return op_spec

    @classmethod
    def register(cls, key, function):
        """
        Register a custom op transform function which can from then
        on be referenced by the key.
        """
        self._op_registry[name] = function

    # Unary operators
    def __abs__(self): return dim(self, operator.abs)
    def __neg__(self): return dim(self, operator.neg)
    def __pos__(self): return dim(self, operator.pos)

    # Binary operators
    def __add__(self, other):       return dim(self, operator.add, other)
    def __div__(self, other):       return dim(self, operator.div, other)
    def __floordiv__(self, other):  return dim(self, operator.floordiv, other)
    def __pow__(self, other):       return dim(self, operator.pow, other)
    def __mod__(self, other):       return dim(self, operator.mod, other)
    def __mul__(self, other):       return dim(self, operator.mul, other)
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

    def max(self, **kwargs):  return dim(self, np.max, **kwargs)
    def mean(self, **kwargs): return dim(self, np.mean, **kwargs)
    def min(self, **kwargs):  return dim(self, np.min, **kwargs)
    def sum(self, **kwargs):  return dim(self, np.sum, **kwargs)
    def std(self, **kwargs):  return dim(self, np.std, **kwargs)
    def var(self, **kwargs):  return dim(self, np.var, **kwargs)
    def astype(self, dtype):  return dim(self, np.asarray, dtype=dtype)

    ## Custom functions

    def norm(self):
        """
        Normalizes the data into the given range
        """
        return dim(self, norm_fn)

    def cat(self, categories, empty=None):
        cat_op = dim(self, cat_fn, categories=categories, empty=empty)
        return cat_op

    def bin(self, bins, labels=None):
        bin_op = dim(self, bin_fn, categories=categories, empty=empty)
        return bin_op

    def eval(self, dataset, flat=False, expanded=None, ranges={}):
        if expanded is None:
            expanded = not ((dataset.interface.gridded and self.dimension in dataset.kdims) or
                            (dataset.interface.multi and dataset.interface.isscalar(dataset, self.dimension)))
        if isinstance(dataset, Graph):
            dataset = dataset if self.dimension in dataset else dataset.nodes
        data = dataset.dimension_values(self.dimension, expanded=expanded, flat=flat)
        for o in self.ops:
            other = o['other']
            if other is not None:
                if isinstance(other, op):
                    other = other.eval(dataset, ranges)
                args = (other, data) if o['reverse'] else (data, other)
            else:
                args = (data,)
            drange = ranges.get(self.dimension.name, {})
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


class norm(op):

    def __init__(self, obj, **kwargs):
        super(norm, self).__init__(obj, norm_fn, **kwargs)

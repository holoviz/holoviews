from __future__ import division

import operator
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, MethodType

import numpy as np

from ..core.dimension import Dimension
from ..core.util import basestring, unique_iterator
from ..element import Graph

function_types = (
    BuiltinFunctionType, BuiltinMethodType, FunctionType,
    MethodType, np.ufunc)


def norm(values, min=None, max=None):
    """Unity-based normalization to scale data into 0-1 range.

        (values - min) / (max - min)

    Args:
        values: Array of values to be normalized
        min (float, optional): Lower bound of normalization range
        max (float, optional): Upper bound of normalization range

    Returns:
        Array of normalized values
    """
    min = np.min(values) if min is None else min
    max = np.max(values) if max is None else max
    return (values - min) / (max-min)


def bin(values, bins, labels=None):
    """Bins data into declared bins

    Bins data into declared bins. By default each bin is labelled
    with bin center values but an explicit list of bin labels may be
    defined.

    Args:
        values: Array of values to be binned
        bins: List or array containing the bin boundaries
        labels: List of labels to assign to each bin
            If the bins are length N the labels should be length N-1

    Returns:
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


def categorize(values, categories, default=None):
    """Maps discrete values to supplied categories.

    Replaces discrete values in input array with a fixed set of
    categories defined either as a list or dictionary.

    Args:
        values: Array of values to be categorized
        categories: List or dict of categories to map inputs to
        default: Default value to assign if value not in categories

    Returns:
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
                cat = default
        else:
            cat = categories.get(c, default)
        cats.append(cat)
    return np.asarray(cats)



class dim(object):
    """
    dim transform objects are a way to express deferred transforms on
    Datasets. dim transforms support all mathematical and bitwise
    operators, NumPy ufuncs and methods, and provide a number of
    useful methods for normalizing, binning and categorizing data.
    """

    _binary_funcs = {
        operator.add: '+', operator.and_: '&', operator.eq: '=',
        operator.floordiv: '//', operator.ge: '>=', operator.gt: '>',
        operator.le: '<=', operator.lshift: '<<', operator.lt: '<',
        operator.mod: '%', operator.mul: '*', operator.ne: '!=',
        operator.or_: '|', operator.pow: '**', operator.rshift: '>>',
        operator.sub: '-', operator.truediv: '/'}

    _builtin_funcs = {abs: 'abs', round: 'round'}

    _custom_funcs = {norm: 'norm', bin: 'bin', categorize: 'categorize'}

    _numpy_funcs = {
        np.any: 'any', np.all: 'all', np.asarray: 'astype',
        np.cumprod: 'cumprod', np.cumsum: 'cumsum', np.max: 'max',
        np.mean: 'mean', np.min: 'min', np.round: 'round',
        np.sum: 'sum', np.std: 'std', np.var: 'var'}

    _unary_funcs = {operator.pos: '+', operator.neg: '-', operator.not_: '~'}

    _all_funcs = [_binary_funcs, _builtin_funcs, _custom_funcs,
                  _numpy_funcs, _unary_funcs]

    _namespaces = {'numpy': 'np'}

    def __init__(self, obj, *args, **kwargs):
        ops = []
        if isinstance(obj, basestring):
            self.dimension = Dimension(obj)
        elif isinstance(obj, Dimension):
            self.dimension = obj
        else:
            self.dimension = obj.dimension
            ops = obj.ops
        if args:
            fn = args[0]
        else:
            fn = None
        if fn is not None:
            if not (isinstance(fn, function_types) or
                    any(fn in funcs for funcs in self._all_funcs)):
                raise ValueError('Second argument must be a function, '
                                 'found %s type' % type(fn))
            ops = ops + [{'args': args[1:], 'fn': fn, 'kwargs': kwargs,
                          'reverse': kwargs.pop('reverse', False)}]
        self.ops = ops

    @classmethod
    def register(cls, key, function):
        """
        Register a custom dim transform function which can from then
        on be referenced by the key.
        """
        cls._custom_funcs[key] = function

    # Builtin functions
    def __abs__(self):            return dim(self, abs)
    def __round__(self, ndigits=None):
        args = () if ndigits is None else (ndigits,)
        return dim(self, round, *args)

    # Unary operators
    def __neg__(self): return dim(self, operator.neg)
    def __not__(self): return dim(self, operator.not_)
    def __pos__(self): return dim(self, operator.pos)

    # Binary operators
    def __add__(self, other):       return dim(self, operator.add, other)
    def __and__(self, other):       return dim(self, operator.and_, other)
    def __div__(self, other):       return dim(self, operator.div, other)
    def __eq__(self, other):        return dim(self, operator.eq, other)
    def __floordiv__(self, other):  return dim(self, operator.floordiv, other)
    def __ge__(self, other):        return dim(self, operator.ge, other)
    def __gt__(self, other):        return dim(self, operator.gt, other)
    def __le__(self, other):        return dim(self, operator.le, other)
    def __lt__(self, other):        return dim(self, operator.lt, other)
    def __lshift__(self, other):    return dim(self, operator.lshift, other)
    def __mod__(self, other):       return dim(self, operator.mod, other)
    def __mul__(self, other):       return dim(self, operator.mul, other)
    def __ne__(self, other):        return dim(self, operator.ne, other)
    def __or__(self, other):        return dim(self, operator.or_, other)
    def __rshift__(self, other):    return dim(self, operator.rshift, other)
    def __pow__(self, other):       return dim(self, operator.pow, other)
    def __sub__(self, other):       return dim(self, operator.sub, other)
    def __truediv__(self, other):   return dim(self, operator.truediv, other)

    # Reverse binary operators
    def __radd__(self, other):      return dim(self, operator.add, other, reverse=True)
    def __rand__(self, other):      return dim(self, operator.and_, other)
    def __rdiv__(self, other):      return dim(self, operator.div, other, reverse=True)
    def __rfloordiv__(self, other): return dim(self, operator.floordiv, other, reverse=True)
    def __rlshift__(self, other):   return dim(self, operator.rlshift, other)
    def __rmod__(self, other):      return dim(self, operator.mod, other, reverse=True)
    def __rmul__(self, other):      return dim(self, operator.mul, other, reverse=True)
    def __ror__(self, other):       return dim(self, operator.or_, other, reverse=True)
    def __rpow__(self, other):      return dim(self, operator.pow, other, reverse=True)
    def __rrshift__(self, other):   return dim(self, operator.rrshift, other)
    def __rsub__(self, other):      return dim(self, operator.sub, other, reverse=True)
    def __rtruediv__(self, other):  return dim(self, operator.truediv, other, reverse=True)

    ## NumPy operations
    def __array_ufunc__(self, *args, **kwargs):
        ufunc = args[0]
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return dim(self, ufunc, **kwargs)

    def clip(self, min=None, max=None):
        if min is None and max is None:
            raise ValueError('One of max or min must be given.')
        return dim(self, np.clip, a_min=min, a_max=max)

    def any(self, **kwargs):     return dim(self, np.any, **kwargs)
    def all(self, **kwargs):     return dim(self, np.all, **kwargs)
    def astype(self, dtype):     return dim(self, np.asarray, dtype=dtype)
    def cumprod(self, **kwargs): return dim(self, np.cumprod, **kwargs)
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
        """Bins continuous values.

        Bins continuous using the provided bins and assigns labels
        either computed from each bins center point or from the
        supplied labels.

        Args:
            bins: List or array containing the bin boundaries
            labels: List of labels to assign to each bin
                If the bins are length N the labels should be length N-1
        """
        return dim(self, bin, bins, labels=labels)

    def categorize(self, categories, default=None):
        """Replaces discrete values with supplied categories

        Replaces discrete values in input array into a fixed set of
        categories defined either as a list or dictionary.

        Args:
            categories: List or dict of categories to map inputs to
            default: Default value to assign if value not in categories
        """
        return dim(self, categorize, categories=categories, default=default)

    def norm(self, limits=None):
        """Unity-based normalization to scale data into 0-1 range.

            (values - min) / (max - min)

        Args:
            limits: tuple of (min, max) defining the normalization range
        """
        kwargs = {}
        if limits is not None:
            kwargs = {'min': limits[0], 'max': limits[1]}
        return dim(self, norm, **kwargs)

    def str(self):
        "Casts values to strings."
        return self.astype(str)

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
            args = op.get('args')
            if not args:
                continue
            for arg in args:
                if isinstance(arg, dim):
                    applies &= arg.applies(dataset)
        return applies

    def apply(self, dataset, flat=False, expanded=None, ranges={}, all_values=False):
        """Evaluates the transform on the supplied dataset.

        Args:
            dataset: Dataset object to evaluate the expression on
            flat: Whether to flatten the returned array
            expanded: Whether to use the expanded expand values
            ranges: Dictionary for ranges for normalization
            all_values: Whether to evaluate on all values
               Whether to evaluate on all available values, for some
               element types, such as Graphs, this may include values
               not included in the referenced column

        Returns:
            values: NumPy array computed by evaluating the expression
        """
        dimension = self.dimension
        if expanded is None:
            expanded = not ((dataset.interface.gridded and dimension in dataset.kdims) or
                            (dataset.interface.multi and dataset.interface.isscalar(dataset, dimension)))
        if isinstance(dataset, Graph):
            if dimension in dataset.kdims and all_values:
                dimension = dataset.nodes.kdims[2]
            dataset = dataset if dimension in dataset else dataset.nodes
        data = dataset.dimension_values(dimension, expanded=expanded, flat=flat)
        for o in self.ops:
            args = o['args']
            fn_args = [data]
            for arg in args:
                if isinstance(arg, dim):
                    arg = arg.apply(dataset, flat, expanded, ranges, all_values)
                fn_args.append(arg)
            args = tuple(fn_args[::-1] if o['reverse'] else fn_args)
            eldim = dataset.get_dimension(dimension)
            drange = ranges.get(eldim.name, {})
            drange = drange.get('combined', drange)
            kwargs = o['kwargs']
            if o['fn'] is norm and drange != {} and not ('min' in kwargs and 'max' in kwargs):
                data = o['fn'](data, *drange)
            else:
                data = o['fn'](*args, **kwargs)
        return data

    def __repr__(self):
        op_repr = "'%s'" % self.dimension
        for o in self.ops:
            if 'dim(' in op_repr:
                prev = '{repr}' if op_repr.endswith(')') else '({repr})'
            else:
                prev = 'dim({repr})'
            fn = o['fn']
            ufunc = isinstance(fn, np.ufunc)
            args = ', '.join([repr(r) for r in o['args']]) if o['args'] else ''
            kwargs = sorted(o['kwargs'].items(), key=operator.itemgetter(0))
            kwargs = '%s' % ', '.join(['%s=%s' % item for item in kwargs]) if kwargs else ''
            if fn in self._binary_funcs:
                fn_name = self._binary_funcs[o['fn']]
                if o['reverse']:
                    format_string = '{args}{fn}'+prev
                else:
                    format_string = prev+'{fn}{args}'
            elif fn in self._unary_funcs:
                fn_name = self._unary_funcs[fn]
                format_string = '{fn}' + prev
            else:
                fn_name = fn.__name__
                if fn in self._builtin_funcs:
                    fn_name = self._builtin_funcs[fn]
                    format_string = '{fn}'+prev
                elif fn in self._numpy_funcs:
                    fn_name = self._numpy_funcs[fn]
                    format_string = prev+'.{fn}('
                elif fn in self._custom_funcs:
                    fn_name = self._custom_funcs[fn]
                    format_string = prev+'.{fn}('
                elif ufunc:
                    fn_name = str(fn)[8:-2]
                    if not (prev.startswith('dim') or prev.endswith(')')):
                        format_string = '{fn}' + prev
                    else:
                        format_string = '{fn}(' + prev
                    if fn_name in dir(np):
                        format_string = '.'.join([self._namespaces['numpy'], format_string])
                else:
                    format_string = prev+', {fn}'
                if args:
                    format_string += ', {args}'
                    if kwargs:
                        format_string += ', {kwargs}'
                elif kwargs:
                    format_string += '{kwargs}'
                format_string += ')'
            op_repr = format_string.format(fn=fn_name, repr=op_repr,
                                           args=args, kwargs=kwargs)
        return op_repr

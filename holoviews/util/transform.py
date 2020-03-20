from __future__ import division

import operator
import sys

from functools import partial
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, MethodType

import numpy as np
import param

from ..core.dimension import Dimension
from ..core.util import basestring, resolve_dependent_value, unique_iterator

def _maybe_map(numpy_fn):
    def fn(values, *args, **kwargs):
        series_like = hasattr(values, 'index') and not isinstance(values, list)
        map_fn = (getattr(values, 'map_partitions', None) or
                  getattr(values, 'map_blocks', None))
        if map_fn:
            if series_like:
                return map_fn(
                    lambda s: type(s)(numpy_fn(s, *args, **kwargs),
                                      index=s.index))
            else:
                return map_fn(lambda s: numpy_fn(s, *args, **kwargs))
        else:
            if series_like:
                return type(values)(
                    numpy_fn(values, *args, **kwargs),
                    index=values.index,
                )
            else:
                return numpy_fn(values, *args, **kwargs)
    return fn


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

def lognorm(values, min=None, max=None):
    """Unity-based normalization on log scale.
       Apply the same transformation as matplotlib.colors.LogNorm

    Args:
        values: Array of values to be normalized
        min (float, optional): Lower bound of normalization range
        max (float, optional): Upper bound of normalization range

    Returns:
        Array of normalized values
    """
    min = np.log(np.min(values)) if min is None else np.log(min)
    max = np.log(np.max(values)) if max is None else np.log(max)
    return (np.log(values) - min) / (max-min)


class iloc(object):
    """Implements integer array indexing for dim expressions.
    """

    __name__ = 'iloc'

    def __init__(self, dim_expr):
        self.expr = dim_expr
        self.index = slice(None)

    def __getitem__(self, index):
        self.index = index
        return dim(self.expr, self)

    def __call__(self, values):
        return values[self.index]


@_maybe_map
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


@_maybe_map
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
    result = np.asarray(cats)
    # Convert unicode to object type like pandas does
    if result.dtype.kind in ['U', 'S']:
        result = result.astype('object')
    return result


digitize = _maybe_map(np.digitize)
isin = _maybe_map(np.isin)
astype = _maybe_map(np.asarray)
round_ = _maybe_map(np.round)

def _python_isin(array, values):
    return [v in values for v in array]

python_isin = _maybe_map(_python_isin)

function_types = (
    BuiltinFunctionType, BuiltinMethodType, FunctionType,
    MethodType, np.ufunc, iloc)


class dim(object):
    """
    dim transform objects are a way to express deferred transforms on
    Datasets. dim transforms support all mathematical and bitwise
    operators, NumPy ufuncs and methods, and provide a number of
    useful methods for normalizing, binning and categorizing data.
    """

    _binary_funcs = {
        operator.add: '+', operator.and_: '&', operator.eq: '==',
        operator.floordiv: '//', operator.ge: '>=', operator.gt: '>',
        operator.le: '<=', operator.lshift: '<<', operator.lt: '<',
        operator.mod: '%', operator.mul: '*', operator.ne: '!=',
        operator.or_: '|', operator.pow: '**', operator.rshift: '>>',
        operator.sub: '-', operator.truediv: '/'}

    _builtin_funcs = {abs: 'abs', round_: 'round'}

    _custom_funcs = {
        norm: 'norm',
        lognorm: 'lognorm',
        bin: 'bin',
        categorize: 'categorize',
        digitize: 'digitize',
        isin: 'isin',
        python_isin: 'isin',
        astype: 'astype',
        round_: 'round',
        iloc: 'iloc'
    }

    _numpy_funcs = {
        np.any: 'any', np.all: 'all',
        np.cumprod: 'cumprod', np.cumsum: 'cumsum', np.max: 'max',
        np.mean: 'mean', np.min: 'min',
        np.sum: 'sum', np.std: 'std', np.var: 'var', np.log: 'log',
        np.log10: 'log10'}

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
            if not (isinstance(fn, function_types+(basestring,)) or
                    any(fn in funcs for funcs in self._all_funcs)):
                raise ValueError('Second argument must be a function, '
                                 'found %s type' % type(fn))
            ops = ops + [{'args': args[1:], 'fn': fn, 'kwargs': kwargs,
                          'reverse': kwargs.pop('reverse', False)}]
        self.ops = ops


    def clone(self, dimension=None, ops=None):
        """
        Creates a clone of the dim expression optionally overriding
        the dim and ops.
        """
        if dimension is None:
            dimension = self.dimension
        new_dim = dim(dimension)
        if ops is None:
            ops = list(self.ops)
        new_dim.ops = ops
        return new_dim

    @classmethod
    def register(cls, key, function):
        """
        Register a custom dim transform function which can from then
        on be referenced by the key.
        """
        cls._custom_funcs[key] = function

    @classmethod
    def pipe(cls, func, *args, **kwargs):
        """
        Wrapper to give multidimensional transforms a more intuitive syntax.
        For a custom function 'func' with signature (*args, **kwargs), call as
        dim.pipe(func, *args, **kwargs).
        """
        args = list(args) # make mutable
        for k, arg in enumerate(args):
            if isinstance(arg, basestring):
                args[k] = dim(arg)
        return dim(args[0], func, *args[1:], **kwargs)

    def __hash__(self):
        return hash(repr(self))

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        return partial(self.method, attr)

    @property
    def params(self):
        params = {}
        for op in self.ops:
            op_args = list(op['args'])+list(op['kwargs'].items())
            for op_arg in op_args:
                if 'panel' in sys.modules:
                    from panel.widgets.base import Widget
                    if isinstance(op_arg, Widget):
                        op_arg = op_arg.param.value
                if (isinstance(op_arg, param.Parameter) and
                    isinstance(op_arg.owner, param.Parameterized)):
                    params[op_arg.name+str(id(op))] = op_arg
        return params

    def method(self, method_str, *args, **kwargs):
        return dim(self, method_str, *args, **kwargs)

    # Builtin functions
    def __abs__(self):            return dim(self, abs)
    def __round__(self, ndigits=None):
        args = () if ndigits is None else (ndigits,)
        return dim(self, round_, *args)

    # Unary operators
    def __neg__(self): return dim(self, operator.neg)
    def __not__(self): return dim(self, operator.not_)
    def __invert__(self): return dim(self, operator.inv)
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

    def any(self, *args, **kwargs):      return dim(self, np.any, *args, **kwargs)
    def all(self, *args, **kwargs):      return dim(self, np.all, *args, **kwargs)
    def cumprod(self, *args, **kwargs):  return dim(self, np.cumprod,  *args, **kwargs)
    def cumsum(self, *args, **kwargs):   return dim(self, np.cumsum,  *args, **kwargs)
    def max(self, *args, **kwargs):      return dim(self, np.max, *args, **kwargs)
    def mean(self, *args, **kwargs):     return dim(self, np.mean, *args, **kwargs)
    def min(self, *args, **kwargs):      return dim(self, np.min, *args, **kwargs)
    def sum(self, *args, **kwargs):      return dim(self, np.sum, *args, **kwargs)
    def std(self, *args, **kwargs):      return dim(self, np.std, *args, **kwargs)
    def var(self, *args, **kwargs):      return dim(self, np.var, *args, **kwargs)
    def log(self, *args, **kwargs):      return dim(self, np.log, *args, **kwargs)
    def log10(self, *args, **kwargs):    return dim(self, np.log10, *args, **kwargs)

    ## Custom functions
    def astype(self, dtype): return dim(self, astype, dtype=dtype)
    def round(self, decimals=0): return dim(self, round_, decimals=decimals)
    def digitize(self, *args, **kwargs): return dim(self, digitize, *args, **kwargs)
    def isin(self, *args, **kwargs):
        if kwargs.pop('object', None):
            return dim(self, python_isin, *args, **kwargs)
        return dim(self, isin, *args, **kwargs)

    @property
    def iloc(self):
        return iloc(self)

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

    def lognorm(self, limits=None):
        """Unity-based normalization log scale.
           Apply the same transformation as matplotlib.colors.LogNorm

        Args:
            limits: tuple of (min, max) defining the normalization range
        """
        kwargs = {}
        if limits is not None:
            kwargs = {'min': limits[0], 'max': limits[1]}
        return dim(self, lognorm, **kwargs)

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
        from ..element import Graph

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

    def apply(self, dataset, flat=False, expanded=None, ranges={}, all_values=False,
              keep_index=False, compute=True, strict=False):
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
           keep_index: For data types that support indexes, whether the index
               should be preserved in the result.
           compute: For data types that support lazy evaluation, whether
               the result should be computed before it is returned.
           strict: Whether to strictly check for dimension matches
               (if False, counts any dimensions with matching names as the same)

        Returns:
            values: NumPy array computed by evaluating the expression
        """
        from ..element import Graph

        dimension = self.dimension
        if expanded is None:
            expanded = not ((dataset.interface.gridded and dimension in dataset.kdims) or
                            (dataset.interface.multi and dataset.interface.isunique(dataset, dimension, True)))

        if isinstance(dataset, Graph):
            if dimension in dataset.kdims and all_values:
                dimension = dataset.nodes.kdims[2]
            dataset = dataset if dimension in dataset else dataset.nodes

        lookup = dimension if strict else dimension.name
        data = dataset.interface.values(
            dataset,
            lookup,
            expanded=expanded,
            flat=flat,
            compute=compute,
            keep_index=keep_index
        )
        for o in self.ops:
            args = o['args']
            fn = o['fn']
            kwargs = dict(o['kwargs'])
            fn_name = self._numpy_funcs.get(fn)
            if fn_name and hasattr(data, fn_name):
                if 'axis' not in kwargs and not isinstance(fn, np.ufunc):
                    kwargs['axis'] = None
                fn = fn_name
            fn_args = [] if isinstance(fn, basestring) else [data]
            for arg in args:
                if isinstance(arg, dim):
                    arg = arg.apply(
                        dataset,
                        flat,
                        expanded,
                        ranges,
                        all_values,
                        keep_index,
                        compute,
                    )
                arg = resolve_dependent_value(arg)
                fn_args.append(arg)
            fn_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, dim):
                    v = v.apply(
                        dataset,
                        flat,
                        expanded,
                        ranges,
                        all_values,
                        keep_index,
                        compute,
                    )
                fn_kwargs[k] = resolve_dependent_value(v)
            args = tuple(fn_args[::-1] if o['reverse'] else fn_args)
            kwargs = dict(fn_kwargs)
            eldim = dataset.get_dimension(lookup)
            drange = ranges.get(eldim.name, {})
            drange = drange.get('combined', drange)
            if (((fn is norm) or (o['fn'] is lognorm)) and drange != {} and
                not ('min' in kwargs and 'max' in kwargs)):
                data = fn(data, *drange)
            elif isinstance(fn, basestring):
                method = getattr(data, fn, None)
                if method is None:
                    raise AttributeError(
                        "%r could not be applied to '%r', '%s' method "
                        "does not exist on %s type."
                        % (self, dataset, fn, type(data).__name__)
                    )
                try:
                    data = method(*args, **kwargs)
                except Exception as e:
                    if 'axis' in kwargs:
                        kwargs.pop('axis')
                        data = method(*args, **kwargs)
                    else:
                        raise e
            else:
                data = fn(*args, **kwargs)
        return data

    def __repr__(self):
        op_repr = "'%s'" % self.dimension
        for i, o in enumerate(self.ops):
            if i == 0:
                prev = 'dim({repr}'
            else:
                prev = '({repr}'
            fn = o['fn']
            ufunc = isinstance(fn, np.ufunc)
            args = ', '.join([repr(r) for r in o['args']]) if o['args'] else ''
            kwargs = sorted(o['kwargs'].items(), key=operator.itemgetter(0))
            kwargs = '%s' % ', '.join(['%s=%r' % item for item in kwargs]) if kwargs else ''
            if fn in self._binary_funcs:
                fn_name = self._binary_funcs[o['fn']]
                if o['reverse']:
                    format_string = '{args}{fn}'+prev
                else:
                    format_string = prev+'){fn}{args}'
                if any(isinstance(a, dim) for a in o['args']):
                    format_string = format_string.replace('{args}', '({args})')
            elif fn in self._unary_funcs:
                fn_name = self._unary_funcs[fn]
                format_string = '{fn}' + prev
            else:
                if isinstance(fn, basestring):
                    fn_name = fn
                else:
                    fn_name = fn.__name__
                if fn in self._builtin_funcs:
                    fn_name = self._builtin_funcs[fn]
                    format_string = '{fn}'+prev
                elif isinstance(fn, basestring):
                    format_string = prev+').{fn}('
                elif fn in self._numpy_funcs:
                    fn_name = self._numpy_funcs[fn]
                    format_string = prev+').{fn}('
                elif isinstance(fn, iloc):
                    format_string = prev+').iloc[{0}]'.format(repr(fn.index))
                elif fn in self._custom_funcs:
                    fn_name = self._custom_funcs[fn]
                    format_string = prev+').{fn}('
                elif ufunc:
                    fn_name = str(fn)[8:-2]
                    if not (prev.startswith('dim') or prev.endswith(')')):
                        format_string = '{fn}' + prev
                    else:
                        format_string = '{fn}(' + prev
                    if fn_name in dir(np):
                        format_string = '.'.join([self._namespaces['numpy'], format_string])
                else:
                    format_string = 'dim(' + prev+', {fn}'
                if args:
                    if not format_string.endswith('('):
                        format_string += ', '
                    format_string += '{args}'
                    if kwargs:
                        format_string += ', {kwargs}'
                elif kwargs:
                    format_string += '{kwargs}'
            op_repr = format_string.format(fn=fn_name, repr=op_repr,
                                           args=args, kwargs=kwargs)
            if op_repr.count('(') - op_repr.count(')') > 0:
                op_repr += ')'
        if not self.ops:
            op_repr = 'dim({repr})'.format(repr=op_repr)
        if op_repr.count('(') - op_repr.count(')') > 0:
            op_repr += ')'
        return op_repr

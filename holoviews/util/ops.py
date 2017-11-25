import operator
import numpy as np

from ..core.dimension import Dimension

def norm_fn(values, min=None, max=None):
    min = np.min(values) if min is None else min
    max = np.max(values) if max is None else max
    return (values - min) / (max-min)
    
class op(object):

    _op_registry = {'norm': norm_fn}

    def __init__(self, obj, fn=None, other=None, reverse=False, **kwargs):
        if isinstance(obj, (str, Dimension)):
            self.dimension = obj
            ops = []
        else:
            self.dimension = obj.dimension
            ops = obj.ops
        if isinstance(fn, str):
            fn = self._op_registry.get(fn)
            if fn is None:
                raise ValueError('Operation function %s not found' % fn)
        if fn is not None:
            ops = ops + [{'other': other, 'fn': fn, 'kwargs': kwargs,
                          'reverse': reverse}]
        self.ops = ops

    # Unary operators
    def __abs__(self): return op(self, operator.abs)
    def __neg__(self): return op(self, operator.neg)
    def __pos__(self): return op(self, operator.pos)

    # Binary operators
    def __add__(self, other):       return op(self, operator.add, other)
    def __div__(self, other):       return op(self, operator.div, other)
    def __floordiv__(self, other):  return op(self, operator.floordiv, other)
    def __pow__(self, other):       return op(self, operator.pow, other)
    def __mod__(self, other):       return op(self, operator.mod, other)
    def __mul__(self, other):       return op(self, operator.mul, other)
    def __sub__(self, other):       return op(self, operator.sub, other)
    def __truediv__(self, other):   return op(self, operator.truediv, other)

    # Reverse binary operators
    def __radd__(self, other):      return op(self, operator.add, other, True)
    def __rdiv__(self, other):      return op(self, operator.div, other, True)
    def __rfloordiv__(self, other): return op(self, operator.floordiv, other, True)
    def __rmod__(self, other):      return op(self, operator.mod, other, True)
    def __rmul__(self, other):      return op(self, operator.mul, other, True)
    def __rsub__(self, other):      return op(self, operator.sub, other, True)
    def __rtruediv__(self, other):  return op(self, operator.truediv, other, True)

    ## NumPy operations
    def __array_ufunc__(self, *args, **kwargs):
        ufunc = getattr(args[0], args[1])
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return op(self, ufunc, **kwargs)
    def max(self, **kwargs): return op(self, np.max, **kwargs)
    def mean(self, **kwargs): return op(self, np.mean, **kwargs)
    def min(self, **kwargs): return op(self, np.min, **kwargs)
    def sum(self, **kwargs): return op(self, np.sum, **kwargs)
    def std(self, **kwargs): return op(self, np.std, **kwargs)
    def std(self, **kwargs): return op(self, np.std, **kwargs)
    def var(self, **kwargs): return op(self, np.var, **kwargs)

    def eval(self, dataset):
        expanded = not (dataset.interface.gridded and self.dimension in dataset.kdims)
        data = dataset.dimension_values(self.dimension, expanded=expanded, flat=False)
        for o in self.ops:
            other = o['other']
            if other is not None:
                if isinstance(other, op):
                    other = other.eval(dataset)
                args = (other, data) if o['reverse'] else (data, other)
            else:
                args = (data,)
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

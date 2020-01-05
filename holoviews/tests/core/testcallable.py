# -*- coding: utf-8 -*-
"""
Unit tests of the Callable object that wraps user callbacks. Also test
how DynamicMap validates and invokes Callable based on its signature.
"""
import param
import sys
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element import Scatter
from holoviews import streams
from holoviews.core.spaces import Callable, Generator, DynamicMap
from holoviews.core.operation import OperationCallable
from holoviews.operation import contours
from functools import partial

from ..utils import LoggingComparisonTestCase

class CallableClass(object):

    @staticmethod
    def somestaticmethod(): pass

    @classmethod
    def someclsmethod(cls): pass

    def __call__(self, *testargs):
        return sum(testargs)


class ParamFunc(param.ParameterizedFunction):

    a = param.Integer(default=1)
    b = param.Number(default=1)

    def __call__(self, a, **params):
        p = param.ParamOverrides(self, params)
        return a * p.b


class TestCallableName(ComparisonTestCase):

    def test_simple_function_name(self):
        def foo(x,y): pass
        self.assertEqual(Callable(foo).name, 'foo')

    def test_simple_lambda_name(self):
        self.assertEqual(Callable(lambda x: x).name, '<lambda>')

    def test_partial_name(self):
        py2match = '<functools.partial object'
        py3match = 'functools.partial('
        match = py2match if sys.version_info < (3,0) else py3match
        cb = Callable(partial(lambda x,y: x, y=4))
        self.assertEqual(cb.name.startswith(match), True)

    def test_generator_expression_name(self):
        if sys.version_info < (3,0):
            cb = Generator((i for i in xrange(10))) # noqa
        else:
            cb = Generator((i for i in range(10)))
        self.assertEqual(cb.name, '<genexpr>')

    def test_generator_name(self):
        def innergen(): yield
        cb = Generator(innergen())
        self.assertEqual(cb.name, 'innergen')

    def test_callable_class_name(self):
        self.assertEqual(Callable(CallableClass()).name, 'CallableClass')

    def test_callable_class_call_method(self):
        self.assertEqual(Callable(CallableClass().__call__).name, 'CallableClass')

    def test_classmethod_name(self):
        self.assertEqual(Callable(CallableClass().someclsmethod).name,
                         'CallableClass.someclsmethod')

    def test_staticmethod_name(self):
        self.assertEqual(Callable(CallableClass().somestaticmethod).name,
                         'somestaticmethod')

    def test_parameterized_fn_name(self):
        self.assertEqual(Callable(ParamFunc).name, 'ParamFunc')

    def test_parameterized_fn_instance_name(self):
        self.assertEqual(Callable(ParamFunc.instance()).name, 'ParamFunc')

    def test_operation_name(self):
        self.assertEqual(Callable(contours).name, 'contours')

    def test_operation_instance_name(self):
        self.assertEqual(Callable(contours.instance()).name, 'contours')

    def test_operation_callable_name(self):
        opcallable = OperationCallable(lambda x: x, operation=contours.instance())
        self.assertEqual(Callable(opcallable).name, 'contours')


class TestSimpleCallableInvocation(LoggingComparisonTestCase):

    def test_callable_fn(self):
        def callback(x): return x
        self.assertEqual(Callable(callback)(3), 3)

    def test_callable_lambda(self):
        self.assertEqual(Callable(lambda x,y: x+y)(3,5), 8)

    def test_callable_lambda_extras(self):
        substr = "Ignoring extra positional argument"
        self.assertEqual(Callable(lambda x,y: x+y)(3,5,10), 8)
        self.log_handler.assertContains('WARNING', substr)

    def test_callable_lambda_extras_kwargs(self):
        substr = "['x'] overriden by keywords"
        self.assertEqual(Callable(lambda x,y: x+y)(3,5,x=10), 15)
        self.log_handler.assertEndsWith('WARNING', substr)

    def test_callable_partial(self):
        self.assertEqual(Callable(partial(lambda x,y: x+y,x=4))(5), 9)

    def test_callable_class(self):
        self.assertEqual(Callable(CallableClass())(1,2,3,4), 10)

    def test_callable_paramfunc(self):
        self.assertEqual(Callable(ParamFunc)(3,b=5), 15)

    def test_callable_paramfunc_instance(self):
        self.assertEqual(Callable(ParamFunc.instance())(3,b=5), 15)


class TestCallableArgspec(ComparisonTestCase):

    def test_callable_fn_argspec(self):
        def callback(x): return x
        self.assertEqual(Callable(callback).argspec.args, ['x'])
        self.assertEqual(Callable(callback).argspec.keywords, None)

    def test_callable_lambda_argspec(self):
        self.assertEqual(Callable(lambda x,y: x+y).argspec.args, ['x','y'])
        self.assertEqual(Callable(lambda x,y: x+y).argspec.keywords, None)

    def test_callable_partial_argspec(self):
        self.assertEqual(Callable(partial(lambda x,y: x+y,x=4)).argspec.args, ['y'])
        self.assertEqual(Callable(partial(lambda x,y: x+y,x=4)).argspec.keywords, None)

    def test_callable_class_argspec(self):
        self.assertEqual(Callable(CallableClass()).argspec.args, [])
        self.assertEqual(Callable(CallableClass()).argspec.keywords, None)
        self.assertEqual(Callable(CallableClass()).argspec.varargs, 'testargs')

    def test_callable_paramfunc_argspec(self):
        self.assertEqual(Callable(ParamFunc).argspec.args, ['a'])
        self.assertEqual(Callable(ParamFunc).argspec.keywords, 'params')
        self.assertEqual(Callable(ParamFunc).argspec.varargs, None)

    def test_callable_paramfunc_instance_argspec(self):
        self.assertEqual(Callable(ParamFunc.instance()).argspec.args, ['a'])
        self.assertEqual(Callable(ParamFunc.instance()).argspec.keywords, 'params')
        self.assertEqual(Callable(ParamFunc.instance()).argspec.varargs, None)


class TestKwargCallableInvocation(ComparisonTestCase):
    """
    Test invocation of Callable with kwargs, even for callbacks with
    positional arguments.
    """

    def test_callable_fn(self):
        def callback(x): return x
        self.assertEqual(Callable(callback)(x=3), 3)

    def test_callable_lambda(self):
        self.assertEqual(Callable(lambda x,y: x+y)(x=3,y=5), 8)

    def test_callable_partial(self):
        self.assertEqual(Callable(partial(lambda x,y: x+y,x=4))(y=5), 9)

    def test_callable_paramfunc(self):
        self.assertEqual(Callable(ParamFunc)(a=3,b=5), 15)


class TestMixedCallableInvocation(ComparisonTestCase):
    """
    Test mixed invocation of Callable with kwargs.
    """

    def test_callable_mixed_1(self):
        def mixed_example(a,b, c=10, d=20):
            return a+b+c+d
        self.assertEqual(Callable(mixed_example)(a=3,b=5), 38)

    def test_callable_mixed_2(self):
        def mixed_example(a,b, c=10, d=20):
            return a+b+c+d
        self.assertEqual(Callable(mixed_example)(3,5,5), 33)


class TestLastArgsKwargs(ComparisonTestCase):

    def test_args_none_before_invocation(self):
        c = Callable(lambda x,y: x+y)
        self.assertEqual(c.args, None)

    def test_kwargs_none_before_invocation(self):
        c = Callable(lambda x,y: x+y)
        self.assertEqual(c.kwargs, None)

    def test_args_invocation(self):
        c = Callable(lambda x,y: x+y)
        c(1,2)
        self.assertEqual(c.args, (1,2))

    def test_kwargs_invocation(self):
        c = Callable(lambda x,y: x+y)
        c(x=1,y=4)
        self.assertEqual(c.kwargs, dict(x=1,y=4))


class TestDynamicMapInvocation(ComparisonTestCase):
    """
    Test that DynamicMap passes kdims and stream parameters correctly to
    Callables.
    """

    def test_dynamic_kdims_only(self):
        def fn(A,B):
            return Scatter([(A,2)], label=A)

        dmap = DynamicMap(fn, kdims=['A','B'])
        self.assertEqual(dmap['Test', 1], Scatter([(1, 2)], label='Test'))

    def test_dynamic_kdims_only_by_position(self):
        def fn(A,B):
            return Scatter([(A,2)], label=A)

        dmap = DynamicMap(fn, kdims=['A-dim','B-dim'])
        self.assertEqual(dmap['Test', 1], Scatter([(1, 2)], label='Test'))

    def test_dynamic_kdims_swapped_by_name(self):
        def fn(A,B):
            return Scatter([(A,2)], label=A)

        dmap = DynamicMap(fn, kdims=['B','A'])
        self.assertEqual(dmap[1,'Test'], Scatter([(1, 2)], label='Test'))


    def test_dynamic_kdims_only_invalid(self):
        def fn(A,B):
            return Scatter([(A,2)], label=A)

        regexp="Callable 'fn' accepts more positional arguments than there are kdims and stream parameters"
        with self.assertRaisesRegexp(KeyError, regexp):
            DynamicMap(fn, kdims=['A'])


    def test_dynamic_kdims_args_only(self):
        def fn(*args):
            (A,B) = args
            return Scatter([(A,2)], label=A)

        dmap = DynamicMap(fn, kdims=['A','B'])
        self.assertEqual(dmap['Test', 1], Scatter([(1, 2)], label='Test'))


    def test_dynamic_streams_only_kwargs(self):
        def fn(x=1, y=2):
            return Scatter([(x,y)], label='default')

        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=[], streams=[xy])
        self.assertEqual(dmap[:], Scatter([(1, 2)], label='default'))


    def test_dynamic_streams_only_keywords(self):
        def fn(**kwargs):
            return Scatter([(kwargs['x'],kwargs['y'])], label='default')

        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=[], streams=[xy])
        self.assertEqual(dmap[:], Scatter([(1, 2)], label='default'))


    def test_dynamic_split_kdims_and_streams(self):
        # Corresponds to the old style of kdims as posargs and streams
        # as kwargs
        def fn(A, x=1, y=2):
            return Scatter([(x,y)], label=A)

        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
        self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))

    def test_dynamic_split_kdims_and_streams_invalid(self):
        # Corresponds to the old style of kdims as posargs and streams
        # as kwargs. Pointeral arg names don't have to match
        def fn(x=1, y=2, B='default'):
            return Scatter([(x,y)], label=B)

        xy = streams.PointerXY(x=1, y=2)
        regexp = "Callback 'fn' signature over (.+?) does not accommodate required kdims"
        with self.assertRaisesRegexp(KeyError, regexp):
            DynamicMap(fn, kdims=['A'], streams=[xy])

    def test_dynamic_split_mismatched_kdims(self):
        # Corresponds to the old style of kdims as posargs and streams
        # as kwargs. Pointeral arg names don't have to match
        def fn(B, x=1, y=2):
            return Scatter([(x,y)], label=B)

        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
        self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))

    def test_dynamic_split_mismatched_kdims_invalid(self):
        # Corresponds to the old style of kdims as posargs and streams
        # as kwargs. Pointeral arg names don't have to match and the
        # stream parameters can be passed by position but *only* if they
        # come first
        def fn(x, y, B):
            return Scatter([(x,y)], label=B)

        xy = streams.PointerXY(x=1, y=2)
        regexp = ("Unmatched positional kdim arguments only allowed "
                  "at the start of the signature")
        with self.assertRaisesRegexp(KeyError, regexp):
            DynamicMap(fn, kdims=['A'], streams=[xy])

    def test_dynamic_split_args_and_kwargs(self):
        # Corresponds to the old style of kdims as posargs and streams
        # as kwargs, captured as *args and **kwargs
        def fn(*args, **kwargs):
            return Scatter([(kwargs['x'],kwargs['y'])], label=args[0])

        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
        self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))


    def test_dynamic_all_keywords(self):
        def fn(A='default', x=1, y=2):
            return Scatter([(x,y)], label=A)

        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
        self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))


    def test_dynamic_keywords_and_kwargs(self):
        def fn(A='default', x=1, y=2, **kws):
            return Scatter([(x,y)], label=A)

        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
        self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))


    def test_dynamic_mixed_kwargs(self):
        def fn(x, A, y):
            return Scatter([(x, y)], label=A)

        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
        self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))



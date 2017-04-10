# -*- coding: utf-8 -*-
"""
Unit tests of the Callable object that wraps user callbacks. Also test
how DynamicMap validates and invokes Callable based on its signature.
"""
import param
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element import Scatter
from holoviews import streams
from holoviews.core.spaces import Callable, DynamicMap
from functools import partial

from . import LoggingComparisonTestCase

class CallableClass(object):

    def __call__(self, *testargs):
        return sum(testargs)


class ParamFunc(param.ParameterizedFunction):

    a = param.Integer(default=1)
    b = param.Number(default=1)

    def __call__(self, a, **params):
        p = param.ParamOverrides(self, params)
        return a * p.b


class TestSimpleCallableInvocation(LoggingComparisonTestCase):

    def setUp(self):
        super(TestSimpleCallableInvocation, self).setUp()

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


class TestDynamicMapInvocation(ComparisonTestCase):
    """
    Test that DynamicMap passes kdims and stream parameters correctly to
    Callables.
    """

    def test_dynamic_kdims_only(self):
        def fn(A,B):
            return Scatter([(A,2)], label=A)

        dmap = DynamicMap(fn, kdims=['A','B'], sampled=True)
        self.assertEqual(dmap['Test', 1], Scatter([(1, 2)], label='Test'))

    def test_dynamic_kdims_only_invalid(self):
        def fn(A,B):
            return Scatter([(A,2)], label=A)

        regexp="Callback positional arguments (.+?) do not accommodate required kdims (.+?)"
        with self.assertRaisesRegexp(KeyError, regexp):
            dmap = DynamicMap(fn, kdims=['A'], sampled=True)


    def test_dynamic_kdims_args_only(self):
        def fn(*args):
            (A,B) = args
            return Scatter([(A,2)], label=A)

        dmap = DynamicMap(fn, kdims=['A','B'], sampled=True)
        self.assertEqual(dmap['Test', 1], Scatter([(1, 2)], label='Test'))


    def test_dynamic_streams_only_kwargs(self):
        def fn(x=1, y=2):
            return Scatter([(x,y)], label='default')

        xy = streams.PositionXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=[], streams=[xy], sampled=True)
        self.assertEqual(dmap[:], Scatter([(1, 2)], label='default'))


    def test_dynamic_streams_only_keywords(self):
        def fn(**kwargs):
            return Scatter([(kwargs['x'],kwargs['y'])], label='default')

        xy = streams.PositionXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=[], streams=[xy], sampled=True)
        self.assertEqual(dmap[:], Scatter([(1, 2)], label='default'))


    def test_dynamic_split_kdims_and_streams(self):
        # Corresponds to the old style of kdims as posargs and streams
        # as kwargs
        def fn(A, x=1, y=2):
            return Scatter([(x,y)], label=A)

        xy = streams.PositionXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy], sampled=True)
        self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))


    def test_dynamic_split_mismatched_kdims_and_streams(self):
        # Corresponds to the old style of kdims as posargs and streams
        # as kwargs. Positional arg names don't have to match
        def fn(B, x=1, y=2):
            return Scatter([(x,y)], label=B)

        xy = streams.PositionXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy], sampled=True)
        self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))


    def test_dynamic_split_args_and_kwargs(self):
        # Corresponds to the old style of kdims as posargs and streams
        # as kwargs, captured as *args and **kwargs
        def fn(*args, **kwargs):
            return Scatter([(kwargs['x'],kwargs['y'])], label=args[0])

        xy = streams.PositionXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy], sampled=True)
        self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))


    def test_dynamic_all_keywords(self):
        def fn(A='default', x=1, y=2):
            return Scatter([(x,y)], label=A)

        xy = streams.PositionXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy], sampled=True)
        self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))


    def test_dynamic_keywords_and_kwargs(self):
        def fn(A='default', x=1, y=2, **kws):
            return Scatter([(x,y)], label=A)

        xy = streams.PositionXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy], sampled=True)
        self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))


    def test_dynamic_mixed_kwargs(self):
        def fn(x, A, y):
            return Scatter([(x, y)], label=A)

        xy = streams.PositionXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy], sampled=True)
        self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))



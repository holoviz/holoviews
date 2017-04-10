# -*- coding: utf-8 -*-
"""
Unit tests of the Callable object that wraps user callbacks
"""
import param
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.spaces import Callable
from functools import partial

from . import LoggingComparisonTestCase

class CallableClass(object):

    def __call__(self, *testargs):
        return sum(testargs)


class ParamFunc(param.ParameterizedFunction):

    a = param.Integer(default=1)
    b = param.Number(default=1)

    def __call__(self, **params):
        p = param.ParamOverrides(self, params)
        return p.a * p.b


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
        self.assertEqual(Callable(ParamFunc)(a=3,b=5), 15)


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
        self.assertEqual(Callable(ParamFunc).argspec.args, [])
        self.assertEqual(Callable(ParamFunc).argspec.keywords, 'params')
        self.assertEqual(Callable(ParamFunc).argspec.varargs, None)


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

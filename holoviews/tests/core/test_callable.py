"""
Unit tests of the Callable object that wraps user callbacks. Also test
how DynamicMap validates and invokes Callable based on its signature.
"""
from functools import partial

import param
import pytest

from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.operation import contours
from holoviews.testing import assert_element_equal

from ..utils import LoggingComparison


class CallableClass:

    @staticmethod
    def somestaticmethod(): pass

    @classmethod
    def someclsmethod(cls): pass

    def someinstancemethod(self, x, y):
        return x + y

    def __call__(self, *testargs):
        return sum(testargs)


class ParamFunc(param.ParameterizedFunction):

    a = param.Integer(default=1)
    b = param.Number(default=1)

    def __call__(self, a, **params):
        p = param.ParamOverrides(self, params)
        return a * p.b


class TestCallableName:

    def test_simple_function_name(self):
        def foo(x,y): pass
        assert Callable(foo).name == 'foo'

    def test_simple_lambda_name(self):
        assert Callable(lambda x: x).name == '<lambda>'

    def test_partial_name(self):
        cb = Callable(partial(lambda x,y: x, y=4))
        assert cb.name.startswith('functools.partial(')

    def test_generator_expression_name(self):
        cb = Generator(i for i in range(10))
        assert cb.name == '<genexpr>'

    def test_generator_name(self):
        def innergen(): yield
        cb = Generator(innergen())
        assert cb.name == 'innergen'

    def test_callable_class_name(self):
        assert Callable(CallableClass()).name == 'CallableClass'

    def test_callable_class_call_method(self):
        assert Callable(CallableClass().__call__).name == 'CallableClass'

    def test_callable_instance_method(self):
        assert Callable(CallableClass().someinstancemethod).name == 'CallableClass.someinstancemethod'

    def test_classmethod_name(self):
        assert Callable(CallableClass().someclsmethod).name == 'CallableClass.someclsmethod'

    def test_staticmethod_name(self):
        assert Callable(CallableClass().somestaticmethod).name == 'somestaticmethod'

    def test_parameterized_fn_name(self):
        assert Callable(ParamFunc).name == 'ParamFunc'

    def test_parameterized_fn_instance_name(self):
        assert Callable(ParamFunc.instance()).name == 'ParamFunc'

    def test_operation_name(self):
        assert Callable(contours).name == 'contours'

    def test_operation_instance_name(self):
        assert Callable(contours.instance()).name == 'contours'

    def test_operation_callable_name(self):
        opcallable = OperationCallable(lambda x: x, operation=contours.instance())
        assert Callable(opcallable).name == 'contours'


class TestSimpleCallableInvocation(LoggingComparison):

    def test_callable_fn(self):
        def callback(x): return x
        assert Callable(callback)(3) == 3

    def test_callable_lambda(self):
        assert Callable(lambda x,y: x+y)(3,5) == 8

    def test_callable_lambda_extras(self):
        substr = "Ignoring extra positional argument"
        assert Callable(lambda x,y: x+y)(3,5,10) == 8
        self.log_handler.assertContains('WARNING', substr)

    def test_callable_lambda_extras_kwargs(self):
        substr = "['x'] overridden by keywords"
        assert Callable(lambda x,y: x+y)(3,5,x=10) == 15
        self.log_handler.assertEndsWith('WARNING', substr)

    def test_callable_partial(self):
        assert Callable(partial(lambda x,y: x+y,x=4))(5) == 9

    def test_callable_class(self):
        assert Callable(CallableClass())(1,2,3,4) == 10

    def test_callable_instance_method(self):
        assert Callable(CallableClass().someinstancemethod)(1, 2) == 3

    def test_callable_partial_instance_method(self):
        assert Callable(partial(CallableClass().someinstancemethod, x=1))(2) == 3

    def test_callable_paramfunc(self):
        assert Callable(ParamFunc)(3,b=5) == 15

    def test_callable_paramfunc_instance(self):
        assert Callable(ParamFunc.instance())(3,b=5) == 15


class TestCallableArgspec:

    def test_callable_fn_argspec(self):
        def callback(x): return x
        assert Callable(callback).argspec.args == ['x']
        assert Callable(callback).argspec.keywords is None

    def test_callable_lambda_argspec(self):
        assert Callable(lambda x,y: x+y).argspec.args == ['x','y']
        assert Callable(lambda x,y: x+y).argspec.keywords is None

    def test_callable_partial_argspec(self):
        assert Callable(partial(lambda x,y: x+y, x=4)).argspec.args == ['y']
        assert Callable(partial(lambda x,y: x+y,x=4)).argspec.keywords is None

    def test_callable_class_argspec(self):
        assert Callable(CallableClass()).argspec.args == []
        assert Callable(CallableClass()).argspec.keywords is None
        assert Callable(CallableClass()).argspec.varargs == 'testargs'

    def test_callable_instance_method(self):
        assert Callable(CallableClass().someinstancemethod).argspec.args == ['x', 'y']
        assert Callable(CallableClass().someinstancemethod).argspec.keywords is None

    def test_callable_partial_instance_method(self):
        assert Callable(partial(CallableClass().someinstancemethod, x=1)).argspec.args == ['y']
        assert Callable(partial(CallableClass().someinstancemethod, x=1)).argspec.keywords is None

    def test_callable_paramfunc_argspec(self):
        assert Callable(ParamFunc).argspec.args == ['a']
        assert Callable(ParamFunc).argspec.keywords == 'params'
        assert Callable(ParamFunc).argspec.varargs is None

    def test_callable_paramfunc_instance_argspec(self):
        assert Callable(ParamFunc.instance()).argspec.args == ['a']
        assert Callable(ParamFunc.instance()).argspec.keywords == 'params'
        assert Callable(ParamFunc.instance()).argspec.varargs is None


class TestKwargCallableInvocation:
    """
    Test invocation of Callable with kwargs, even for callbacks with
    positional arguments.
    """

    def test_callable_fn(self):
        def callback(x): return x
        assert Callable(callback)(x=3) == 3

    def test_callable_lambda(self):
        assert Callable(lambda x,y: x+y)(x=3,y=5) == 8

    def test_callable_partial(self):
        assert Callable(partial(lambda x,y: x+y,x=4))(y=5) == 9

    def test_callable_instance_method(self):
        assert Callable(CallableClass().someinstancemethod)(x=1, y=2) == 3

    def test_callable_partial_instance_method(self):
        assert Callable(partial(CallableClass().someinstancemethod, x=1))(y=2) == 3

    def test_callable_paramfunc(self):
        assert Callable(ParamFunc)(a=3,b=5) == 15


class TestMixedCallableInvocation:
    """
    Test mixed invocation of Callable with kwargs.
    """

    def test_callable_mixed_1(self):
        def mixed_example(a,b, c=10, d=20):
            return a+b+c+d
        assert Callable(mixed_example)(a=3,b=5) == 38

    def test_callable_mixed_2(self):
        def mixed_example(a,b, c=10, d=20):
            return a+b+c+d
        assert Callable(mixed_example)(3,5,5) == 33


class TestLastArgsKwargs:

    def test_args_none_before_invocation(self):
        c = Callable(lambda x,y: x+y)
        assert c.args is None

    def test_kwargs_none_before_invocation(self):
        c = Callable(lambda x,y: x+y)
        assert c.kwargs is None

    def test_args_invocation(self):
        c = Callable(lambda x,y: x+y)
        c(1,2)
        assert c.args == (1,2)

    def test_kwargs_invocation(self):
        c = Callable(lambda x,y: x+y)
        c(x=1,y=4)
        assert c.kwargs == dict(x=1,y=4)


class TestDynamicMapInvocation:
    """
    Test that DynamicMap passes kdims and stream parameters correctly to
    Callables.
    """

    def test_dynamic_kdims_only(self):
        def fn(A,B):
            return Scatter([(B,2)], label=A)

        dmap = DynamicMap(fn, kdims=['A','B'])
        assert_element_equal(dmap['Test', 1], Scatter([(1, 2)], label='Test'))

    def test_dynamic_kdims_only_by_position(self):
        def fn(A,B):
            return Scatter([(B,2)], label=A)

        dmap = DynamicMap(fn, kdims=['A-dim','B-dim'])
        assert_element_equal(dmap['Test', 1], Scatter([(1, 2)], label='Test'))

    def test_dynamic_kdims_swapped_by_name(self):
        def fn(A,B):
            return Scatter([(B,2)], label=A)

        dmap = DynamicMap(fn, kdims=['B','A'])
        assert_element_equal(dmap[1,'Test'], Scatter([(1, 2)], label='Test'))


    def test_dynamic_kdims_only_invalid(self):
        def fn(A,B):
            return Scatter([(B,2)], label=A)

        regexp="Callable 'fn' accepts more positional arguments than there are kdims and stream parameters"
        with pytest.raises(KeyError, match=regexp):
            DynamicMap(fn, kdims=['A'])


    def test_dynamic_kdims_args_only(self):
        def fn(*args):
            (A,B) = args
            return Scatter([(B,2)], label=A)

        dmap = DynamicMap(fn, kdims=['A','B'])
        assert_element_equal(dmap['Test', 1], Scatter([(1, 2)], label='Test'))


    def test_dynamic_streams_only_kwargs(self):
        def fn(x=1, y=2):
            return Scatter([(x,y)], label='default')

        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=[], streams=[xy])
        assert_element_equal(dmap[:], Scatter([(1, 2)], label='default'))


    def test_dynamic_streams_only_keywords(self):
        def fn(**kwargs):
            return Scatter([(kwargs['x'],kwargs['y'])], label='default')

        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=[], streams=[xy])
        assert_element_equal(dmap[:], Scatter([(1, 2)], label='default'))


    def test_dynamic_split_kdims_and_streams(self):
        # Corresponds to the old style of kdims as posargs and streams
        # as kwargs
        def fn(A, x=1, y=2):
            return Scatter([(x,y)], label=A)

        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
        assert_element_equal(dmap['Test'], Scatter([(1, 2)], label='Test'))

    def test_dynamic_split_kdims_and_streams_invalid(self):
        # Corresponds to the old style of kdims as posargs and streams
        # as kwargs. Pointeral arg names don't have to match
        def fn(x=1, y=2, B='default'):
            return Scatter([(x,y)], label=B)

        xy = streams.PointerXY(x=1, y=2)
        regexp = "Callback 'fn' signature over (.+?) does not accommodate required kdims"
        with pytest.raises(KeyError, match=regexp):
            DynamicMap(fn, kdims=['A'], streams=[xy])

    def test_dynamic_split_mismatched_kdims(self):
        # Corresponds to the old style of kdims as posargs and streams
        # as kwargs. Pointeral arg names don't have to match
        def fn(B, x=1, y=2):
            return Scatter([(x,y)], label=B)

        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
        assert_element_equal(dmap['Test'], Scatter([(1, 2)], label='Test'))

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
        with pytest.raises(KeyError, match=regexp):
            DynamicMap(fn, kdims=['A'], streams=[xy])

    def test_dynamic_split_args_and_kwargs(self):
        # Corresponds to the old style of kdims as posargs and streams
        # as kwargs, captured as *args and **kwargs
        def fn(*args, **kwargs):
            return Scatter([(kwargs['x'],kwargs['y'])], label=args[0])

        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
        assert_element_equal(dmap['Test'], Scatter([(1, 2)], label='Test'))


    def test_dynamic_all_keywords(self):
        def fn(A='default', x=1, y=2):
            return Scatter([(x,y)], label=A)

        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
        assert_element_equal(dmap['Test'], Scatter([(1, 2)], label='Test'))


    def test_dynamic_keywords_and_kwargs(self):
        def fn(A='default', x=1, y=2, **kws):
            return Scatter([(x,y)], label=A)

        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
        assert_element_equal(dmap['Test'], Scatter([(1, 2)], label='Test'))


    def test_dynamic_mixed_kwargs(self):
        def fn(x, A, y):
            return Scatter([(x, y)], label=A)

        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
        assert_element_equal(dmap['Test'], Scatter([(1, 2)], label='Test'))

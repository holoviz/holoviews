import numpy as np
from holoviews import Dimension, DynamicMap, Image, HoloMap
from holoviews.core.operation import DynamicFunction
from holoviews.element.comparison import ComparisonTestCase

frequencies =  np.linspace(0.5,2.0,5)
phases = np.linspace(0, np.pi*2, 5)
x,y = np.mgrid[-5:6, -5:6] * 0.1

def sine_array(phase, freq):
    return np.sin(phase + (freq*x**2+freq*y**2))



class DynamicTestGeneratorOpen(ComparisonTestCase):

    def test_generator_open_init(self):
        generator = (Image(sine_array(0,i)) for i in range(10))
        dmap=DynamicMap(generator)
        self.assertEqual(dmap.mode, 'open')

    def test_generator_open_clone(self):
        generator = (Image(sine_array(0,i)) for i in range(10))
        dmap=DynamicMap(generator)
        self.assertEqual(dmap, dmap.clone())

    def test_generator_open_stopiteration(self):
        generator = (Image(sine_array(0,i)) for i in range(10))
        dmap=DynamicMap(generator)
        for i in range(10):
            el = next(dmap)
            self.assertEqual(type(el), Image)
        try:
            el = next(dmap)
            raise AssertionError("StopIteration not raised when expected")
        except Exception as e:
            if e.__class__ != StopIteration:
                raise AssertionError("StopIteration was expected, got %s" % e)



class DynamicTestCallableOpen(ComparisonTestCase):

    def test_callable_open_init(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn)
        self.assertEqual(dmap.mode, 'open')

    def test_callable_open_clone(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn)
        self.assertEqual(dmap, dmap.clone())




class DynamicTestCallableBounded(ComparisonTestCase):

    def test_callable_bounded_init(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, kdims=[Dimension('dim', range=(0,10))])
        self.assertEqual(dmap.mode, 'bounded')

    def test_generator_bounded_clone(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, kdims=[Dimension('dim', range=(0,10))])
        self.assertEqual(dmap, dmap.clone())


class DynamicTestSampledBounded(ComparisonTestCase):

    def test_sampled_bounded_init(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, sampled=True)
        self.assertEqual(dmap.mode, 'bounded')

    def test_sampled_bounded_resample(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, sampled=True)
        self.assertEqual(dmap[{0, 1, 2}].keys(), [0, 1, 2])


class DynamicTestOperation(ComparisonTestCase):

    def test_dynamic_function(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, sampled=True)
        dmap_with_fn = DynamicFunction(dmap, function=lambda x: x.clone(x.data*2))
        self.assertEqual(dmap_with_fn[5], Image(sine_array(0,5)*2))


    def test_dynamic_function_with_kwargs(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, sampled=True)
        def fn(x, multiplier=2):
            return x.clone(x.data*multiplier)
        dmap_with_fn = DynamicFunction(dmap, function=fn, kwargs=dict(multiplier=3))
        self.assertEqual(dmap_with_fn[5], Image(sine_array(0,5)*3))



class DynamicTestOverlay(ComparisonTestCase):

    def test_dynamic_element_overlay(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, sampled=True)
        dynamic_overlay = dmap * Image(sine_array(0,10))
        overlaid = Image(sine_array(0,5)) * Image(sine_array(0,10))
        self.assertEqual(dynamic_overlay[5], overlaid)

    def test_dynamic_element_underlay(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, sampled=True)
        dynamic_overlay = Image(sine_array(0,10)) * dmap
        overlaid = Image(sine_array(0,10)) * Image(sine_array(0,5))
        self.assertEqual(dynamic_overlay[5], overlaid)

    def test_dynamic_dynamicmap_overlay(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, sampled=True)
        fn2 = lambda i: Image(sine_array(0,i*2))
        dmap2=DynamicMap(fn2, sampled=True)
        dynamic_overlay = dmap * dmap2
        overlaid = Image(sine_array(0,5)) * Image(sine_array(0,10))
        self.assertEqual(dynamic_overlay[5], overlaid)

    def test_dynamic_holomap_overlay(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap = DynamicMap(fn, sampled=True)
        hmap = HoloMap({i: Image(sine_array(0,i*2)) for i in range(10)})
        dynamic_overlay = dmap * hmap
        overlaid = Image(sine_array(0,5)) * Image(sine_array(0,10))
        self.assertEqual(dynamic_overlay[5], overlaid)

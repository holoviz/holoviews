import numpy as np
from holoviews import Dimension, DynamicMap, Image, HoloMap, Scatter, Curve
from holoviews.util import Dynamic
from holoviews.element.comparison import ComparisonTestCase

frequencies =  np.linspace(0.5,2.0,5)
phases = np.linspace(0, np.pi*2, 5)
x,y = np.mgrid[-5:6, -5:6] * 0.1

def sine_array(phase, freq):
    return np.sin(phase + (freq*x**2+freq*y**2))


class DynamicMethods(ComparisonTestCase):

    def test_deep_relabel_label(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap = DynamicMap(fn).relabel(label='Test')
        self.assertEqual(dmap[0].label, 'Test')

    def test_deep_relabel_group(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap = DynamicMap(fn).relabel(group='Test')
        self.assertEqual(dmap[0].group, 'Test')

    def test_redim_dimension_name(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap = DynamicMap(fn).redim(Default='New')
        self.assertEqual(dmap.kdims[0].name, 'New')

    def test_deep_redim_dimension_name(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap = DynamicMap(fn).redim(x='X')
        self.assertEqual(dmap[0].kdims[0].name, 'X')

    def test_deep_redim_dimension_name_with_spec(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap = DynamicMap(fn).redim(Image, x='X')
        self.assertEqual(dmap[0].kdims[0].name, 'X')

    def test_deep_getitem_bounded_kdims(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap[:, 5:10][10], fn(10)[5:10])

    def test_deep_getitem_bounded_kdims_and_vdims(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap[:, 5:10, 0:5][10], fn(10)[5:10, 0:5])

    def test_deep_getitem_cross_product_and_slice(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap[[10, 11, 12], 5:10],
                         dmap.clone([(i, fn(i)[5:10]) for i in range(10, 13)]))

    def test_deep_getitem_index_and_slice(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap[10, 5:10], fn(10)[5:10])

    def test_deep_getitem_cache_sliced(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        dmap[10] # Add item to cache
        self.assertEqual(dmap[:, 5:10][10], fn(10)[5:10])

    def test_deep_select_slice_kdim(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap.select(x=(5, 10))[10], fn(10)[5:10])

    def test_deep_select_slice_kdim_and_vdims(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap.select(x=(5, 10), y=(0, 5))[10], fn(10)[5:10, 0:5])

    def test_deep_select_slice_kdim_no_match(self):
        fn = lambda i: Curve(np.arange(i))
        dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
        self.assertEqual(dmap.select(DynamicMap, x=(5, 10))[10], fn(10))




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

    def test_dynamic_operation(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, sampled=True)
        dmap_with_fn = Dynamic(dmap, operation=lambda x: x.clone(x.data*2))
        self.assertEqual(dmap_with_fn[5], Image(sine_array(0,5)*2))


    def test_dynamic_operation_with_kwargs(self):
        fn = lambda i: Image(sine_array(0,i))
        dmap=DynamicMap(fn, sampled=True)
        def fn(x, multiplier=2):
            return x.clone(x.data*multiplier)
        dmap_with_fn = Dynamic(dmap, operation=fn, kwargs=dict(multiplier=3))
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

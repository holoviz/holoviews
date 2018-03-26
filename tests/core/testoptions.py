import os
import pickle
import numpy as np
from holoviews import Store, StoreOptions, Histogram, Image
from holoviews.core.options import OptionError, Cycle, Options, OptionTree, options_policy
from holoviews.element.comparison import ComparisonTestCase
from holoviews import plotting              # noqa Register backends
from unittest import SkipTest
from nose.plugins.attrib import attr

Options.skip_invalid = False

try:
    # Needed a backend to register backend and options
    from holoviews.plotting import mpl # noqa
except:
    pass

try:
    # Needed to register backend  and options
    from holoviews.plotting import bokeh # noqa
except:
    pass

class TestOptions(ComparisonTestCase):

    def test_options_init(self):
        Options('test')

    def test_options_valid_keywords1(self):
        opts = Options('test', allowed_keywords=['kw1'], kw1='value')
        self.assertEquals(opts.kwargs, {'kw1':'value'})

    def test_options_valid_keywords2(self):
        opts = Options('test', allowed_keywords=['kw1', 'kw2'], kw1='value')
        self.assertEquals(opts.kwargs, {'kw1':'value'})

    def test_options_valid_keywords3(self):
        opts = Options('test', allowed_keywords=['kw1', 'kw2'], kw1='value1', kw2='value2')
        self.assertEquals(opts.kwargs, {'kw1':'value1', 'kw2':'value2'})

    def test_options_any_keywords3(self):
        opts = Options('test', kw1='value1', kw2='value3')
        self.assertEquals(opts.kwargs, {'kw1':'value1', 'kw2':'value3'})

    def test_options_invalid_keywords1(self):
        try:
            Options('test', allowed_keywords=['kw1'], kw='value')
        except OptionError as e:
            self.assertEqual(str(e), "Invalid option 'kw', valid options are: ['kw1']")

    def test_options_invalid_keywords2(self):
        try:
            Options('test', allowed_keywords=['kw2'], kw2='value', kw3='value')
        except OptionError as e:
            self.assertEqual(str(e), "Invalid option 'kw3', valid options are: ['kw2']")

    def test_options_invalid_keywords_skip1(self):
        with options_policy(skip_invalid=True, warn_on_skip=False):
            opts = Options('test', allowed_keywords=['kw1'], kw='value')
        self.assertEqual(opts.kwargs, {})

    def test_options_invalid_keywords_skip2(self):
        with options_policy(skip_invalid=True, warn_on_skip=False):
            opts = Options('test', allowed_keywords=['kw1'], kw1='value', kw2='val')
        self.assertEqual(opts.kwargs, {'kw1':'value'})


    def test_options_record_invalid(self):
        StoreOptions.start_recording_skipped()
        with options_policy(skip_invalid=True, warn_on_skip=False):
            Options('test', allowed_keywords=['kw1'], kw1='value', kw2='val')
        errors = StoreOptions.stop_recording_skipped()
        self.assertEqual(len(errors),1)
        self.assertEqual(errors[0].invalid_keyword,'kw2')


    def test_options_get_options(self):
        opts = Options('test', allowed_keywords=['kw2', 'kw3'],
                       kw2='value', kw3='value').options
        self.assertEqual(opts, dict(kw2='value', kw3='value'))

    def test_options_get_options_cyclic1(self):
        opts = Options('test', allowed_keywords=['kw2', 'kw3'],
                       kw2='value', kw3='value')
        for i in range(16):
            self.assertEqual(opts[i], dict(kw2='value', kw3='value'))

    def test_options_keys(self):
        opts = Options('test', allowed_keywords=['kw3', 'kw2'],
                       kw2='value', kw3='value')
        self.assertEqual(opts.keys(), ['kw2', 'kw3'])

    def test_options_inherit(self):
        original_kws = dict(kw2='value', kw3='value')
        opts = Options('test', **original_kws)
        new_kws = dict(kw4='val4', kw5='val5')
        new_opts = opts(**new_kws)

        self.assertEqual(new_opts.options, dict(original_kws, **new_kws))

    def test_options_inherit_invalid_keywords(self):
        original_kws = dict(kw2='value', kw3='value')
        opts = Options('test', allowed_keywords=['kw3', 'kw2'], **original_kws)
        new_kws = dict(kw4='val4', kw5='val5')
        try:
            opts(**new_kws)
        except OptionError as e:
            self.assertEqual(str(e), "Invalid option 'kw4', valid options are: ['kw2', 'kw3']")



class TestCycle(ComparisonTestCase):

    def test_cycle_init(self):
        Cycle(values=['a', 'b', 'c'])
        Cycle(values=[1, 2, 3])


    def test_cycle_expansion(self):
        cycle1 = Cycle(values=['a', 'b', 'c'])
        cycle2 = Cycle(values=[1, 2, 3])

        opts = Options('test', one=cycle1, two=cycle2)
        self.assertEqual(opts[0], {'one': 'a', 'two': 1})
        self.assertEqual(opts[1], {'one': 'b', 'two': 2})
        self.assertEqual(opts[2], {'one': 'c', 'two': 3})
        self.assertEqual(opts[3], {'one': 'a', 'two': 1})
        self.assertEqual(opts[4], {'one': 'b', 'two': 2})
        self.assertEqual(opts[5], {'one': 'c', 'two': 3})


    def test_cycle_expansion_unequal(self):
        cycle1 = Cycle(values=['a', 'b', 'c', 'd'])
        cycle2 = Cycle(values=[1, 2, 3])

        opts = Options('test', one=cycle1, two=cycle2)
        self.assertEqual(opts[0], {'one': 'a', 'two': 1})
        self.assertEqual(opts[1], {'one': 'b', 'two': 2})
        self.assertEqual(opts[2], {'one': 'c', 'two': 3})
        self.assertEqual(opts[3], {'one': 'd', 'two': 1})
        self.assertEqual(opts[4], {'one': 'a', 'two': 2})
        self.assertEqual(opts[5], {'one': 'b', 'two': 3})


    def test_cycle_slice(self):
        cycle1 = Cycle(values=['a', 'b', 'c'])[2]
        cycle2 = Cycle(values=[1, 2, 3])

        opts = Options('test', one=cycle1, two=cycle2)
        self.assertEqual(opts[0], {'one': 'a', 'two': 1})
        self.assertEqual(opts[1], {'one': 'b', 'two': 2})
        self.assertEqual(opts[2], {'one': 'a', 'two': 3})
        self.assertEqual(opts[3], {'one': 'b', 'two': 1})


    def test_cyclic_property_true(self):
        cycle1 = Cycle(values=['a', 'b', 'c'])
        opts = Options('test', one=cycle1, two='two')
        self.assertEqual(opts.cyclic, True)

    def test_cyclic_property_false(self):
        opts = Options('test', one='one', two='two')
        self.assertEqual(opts.cyclic, False)


    def test_options_property_disabled(self):
        cycle1 = Cycle(values=['a', 'b', 'c'])
        opts = Options('test', one=cycle1)
        try:
            opts.options
        except Exception as e:
            self.assertEqual(str(e), "The options property may only be used with non-cyclic Options.")



class TestOptionTree(ComparisonTestCase):

    def test_optiontree_init_1(self):
        OptionTree(groups=['group1', 'group2'])

    def test_optiontree_init_2(self):
        OptionTree(groups=['group1', 'group2'])

    def test_optiontree_setter_getter(self):
        options = OptionTree(groups=['group1', 'group2'])
        opts = Options('group1', kw1='value')
        options.MyType = opts
        self.assertEqual(options.MyType['group1'], opts)
        self.assertEqual(options.MyType['group1'].options, {'kw1':'value'})

    def test_optiontree_dict_setter_getter(self):
        options = OptionTree(groups=['group1', 'group2'])

        opts1 = Options(kw1='value1')
        opts2 = Options(kw2='value2')
        options.MyType = {'group1':opts1, 'group2':opts2}

        self.assertEqual(options.MyType['group1'], opts1)
        self.assertEqual(options.MyType['group1'].options, {'kw1':'value1'})

        self.assertEqual(options.MyType['group2'], opts2)
        self.assertEqual(options.MyType['group2'].options, {'kw2':'value2'})

    def test_optiontree_inheritance(self):
        if 'matplotlib' not in Store.renderers:
            raise SkipTest("General to specific option test requires matplotlib")

        options = OptionTree(groups=['group1', 'group2'])

        opts1 = Options(kw1='value1')
        opts2 = Options(kw2='value2')
        options.MyType = {'group1':opts1, 'group2':opts2}

        opts3 = Options(kw3='value3')
        opts4 = Options(kw4='value4')
        options.MyType.Child = {'group1':opts3, 'group2':opts4}

        self.assertEqual(options.MyType.Child.options('group1').kwargs,
                         {'kw1':'value1', 'kw3':'value3'})

        self.assertEqual(options.MyType.Child.options('group2').kwargs,
                         {'kw2':'value2', 'kw4':'value4'})

    def test_optiontree_inheritance_flipped(self):
        """
        Tests for ordering problems manifested in issue #93
        """
        if 'matplotlib' not in Store.renderers:
            raise SkipTest("General to specific option test requires matplotlib")

        options = OptionTree(groups=['group1', 'group2'])

        opts3 = Options(kw3='value3')
        opts4 = Options(kw4='value4')
        options.MyType.Child = {'group1':opts3, 'group2':opts4}

        opts1 = Options(kw1='value1')
        opts2 = Options(kw2='value2')
        options.MyType = {'group1':opts1, 'group2':opts2}

        self.assertEqual(options.MyType.Child.options('group1').kwargs,
                         {'kw1':'value1', 'kw3':'value3'})

        self.assertEqual(options.MyType.Child.options('group2').kwargs,
                         {'kw2':'value2', 'kw4':'value4'})


@attr(optional=1) # Requires matplotlib
class TestStoreInheritanceDynamic(ComparisonTestCase):
    """
    Tests to prevent regression after fix in PR #646
    """

    def setUp(self):
        self.store_copy = OptionTree(sorted(Store.options().items()),
                                     groups=['style', 'plot', 'norm'])
        self.backend = 'matplotlib'
        Store.current_backend = self.backend
        super(TestStoreInheritanceDynamic, self).setUp()

    def tearDown(self):
        Store.options(val=self.store_copy)
        Store._custom_options = {k:{} for k in Store._custom_options.keys()}
        super(TestStoreInheritanceDynamic, self).tearDown()

    def initialize_option_tree(self):
        Store.options(val=OptionTree(groups=['plot', 'style']))
        options = Store.options()
        options.Image = Options('style', cmap='hot', interpolation='nearest')
        return options

    def test_merge_keywords(self):
        options = self.initialize_option_tree()
        options.Image = Options('style', clims=(0, 0.5))

        expected = {'clims': (0, 0.5), 'cmap': 'hot', 'interpolation': 'nearest'}
        direct_kws = options.Image.groups['style'].kwargs
        inherited_kws = options.Image.options('style').kwargs
        self.assertEqual(direct_kws, expected)
        self.assertEqual(inherited_kws, expected)

    def test_merge_keywords_disabled(self):
        options = self.initialize_option_tree()
        options.Image = Options('style', clims=(0, 0.5), merge_keywords=False)

        expected = {'clims': (0, 0.5)}
        direct_kws = options.Image.groups['style'].kwargs
        inherited_kws = options.Image.options('style').kwargs
        self.assertEqual(direct_kws, expected)
        self.assertEqual(inherited_kws, expected)

    def test_specification_general_to_specific_group(self):
        """
        Test order of specification starting with general and moving
        to specific
        """
        if 'matplotlib' not in Store.renderers:
            raise SkipTest("General to specific option test requires matplotlib")

        options = self.initialize_option_tree()

        obj = Image(np.random.rand(10,10), group='SomeGroup')

        options.Image = Options('style', cmap='viridis')
        options.Image.SomeGroup = Options('style', alpha=0.2)

        expected = {'alpha': 0.2, 'cmap': 'viridis', 'interpolation': 'nearest'}
        lookup = Store.lookup_options('matplotlib', obj, 'style')

        self.assertEqual(lookup.kwargs, expected)
        # Check the tree is structured as expected
        node1 = options.Image.groups['style']
        node2 = options.Image.SomeGroup.groups['style']

        self.assertEqual(node1.kwargs, {'cmap': 'viridis', 'interpolation': 'nearest'})
        self.assertEqual(node2.kwargs, {'alpha': 0.2})


    def test_specification_general_to_specific_group_and_label(self):
        """
        Test order of specification starting with general and moving
        to specific
        """
        if 'matplotlib' not in Store.renderers:
            raise SkipTest("General to specific option test requires matplotlib")

        options = self.initialize_option_tree()

        obj = Image(np.random.rand(10,10), group='SomeGroup', label='SomeLabel')

        options.Image = Options('style', cmap='viridis')
        options.Image.SomeGroup.SomeLabel = Options('style', alpha=0.2)

        expected = {'alpha': 0.2, 'cmap': 'viridis', 'interpolation': 'nearest'}
        lookup = Store.lookup_options('matplotlib', obj, 'style')

        self.assertEqual(lookup.kwargs, expected)
        # Check the tree is structured as expected
        node1 = options.Image.groups['style']
        node2 = options.Image.SomeGroup.SomeLabel.groups['style']

        self.assertEqual(node1.kwargs, {'cmap': 'viridis', 'interpolation': 'nearest'})
        self.assertEqual(node2.kwargs, {'alpha': 0.2})

    def test_specification_specific_to_general_group(self):
        """
        Test order of specification starting with a specific option and
        then specifying a general one
        """
        if 'matplotlib' not in Store.renderers:
            raise SkipTest("General to specific option test requires matplotlib")

        options = self.initialize_option_tree()
        options.Image.SomeGroup = Options('style', alpha=0.2)

        obj = Image(np.random.rand(10,10), group='SomeGroup')
        options.Image = Options('style', cmap='viridis')

        expected = {'alpha': 0.2, 'cmap': 'viridis', 'interpolation': 'nearest'}
        lookup = Store.lookup_options('matplotlib', obj, 'style')

        self.assertEqual(lookup.kwargs, expected)
        # Check the tree is structured as expected
        node1 = options.Image.groups['style']
        node2 = options.Image.SomeGroup.groups['style']

        self.assertEqual(node1.kwargs, {'cmap': 'viridis', 'interpolation': 'nearest'})
        self.assertEqual(node2.kwargs, {'alpha': 0.2})


    def test_specification_specific_to_general_group_and_label(self):
        """
        Test order of specification starting with general and moving
        to specific
        """
        if 'matplotlib' not in Store.renderers:
            raise SkipTest("General to specific option test requires matplotlib")

        options = self.initialize_option_tree()
        options.Image.SomeGroup.SomeLabel = Options('style', alpha=0.2)
        obj = Image(np.random.rand(10,10), group='SomeGroup', label='SomeLabel')

        options.Image = Options('style', cmap='viridis')
        expected = {'alpha': 0.2, 'cmap': 'viridis', 'interpolation': 'nearest'}
        lookup = Store.lookup_options('matplotlib', obj, 'style')

        self.assertEqual(lookup.kwargs, expected)
        # Check the tree is structured as expected
        node1 = options.Image.groups['style']
        node2 = options.Image.SomeGroup.SomeLabel.groups['style']

        self.assertEqual(node1.kwargs, {'cmap': 'viridis', 'interpolation': 'nearest'})
        self.assertEqual(node2.kwargs, {'alpha': 0.2})

    def test_custom_call_to_default_inheritance(self):
        """
        Checks customs inheritance backs off to default tree correctly
        using __call__.
        """
        options = self.initialize_option_tree()
        options.Image.A.B = Options('style', alpha=0.2)

        obj = Image(np.random.rand(10, 10), group='A', label='B')
        expected_obj =  {'alpha': 0.2, 'cmap': 'hot', 'interpolation': 'nearest'}
        obj_lookup = Store.lookup_options('matplotlib', obj, 'style')
        self.assertEqual(obj_lookup.kwargs, expected_obj)

        # Customize this particular object
        custom_obj = obj(style=dict(clims=(0, 0.5)))
        expected_custom_obj =  dict(clims=(0,0.5), **expected_obj)
        custom_obj_lookup = Store.lookup_options('matplotlib', custom_obj, 'style')
        self.assertEqual(custom_obj_lookup.kwargs, expected_custom_obj)

    def test_custom_magic_to_default_inheritance(self):
        """
        Checks customs inheritance backs off to default tree correctly
        simulating the %%opts cell magic.
        """
        if 'matplotlib' not in Store.renderers:
            raise SkipTest("Custom magic inheritance test requires matplotlib")
        options = self.initialize_option_tree()
        options.Image.A.B = Options('style', alpha=0.2)

        obj = Image(np.random.rand(10, 10), group='A', label='B')

        # Before customizing...
        expected_obj =  {'alpha': 0.2, 'cmap': 'hot', 'interpolation': 'nearest'}
        obj_lookup = Store.lookup_options('matplotlib', obj, 'style')
        self.assertEqual(obj_lookup.kwargs, expected_obj)

        custom_tree = {0: OptionTree(groups=['plot', 'style', 'norm'],
                                     style={'Image' : dict(clims=(0, 0.5))})}
        Store._custom_options['matplotlib'] = custom_tree
        obj.id = 0 # Manually set the id to point to the tree above

        # Customize this particular object
        expected_custom_obj =  dict(clims=(0,0.5), **expected_obj)
        custom_obj_lookup = Store.lookup_options('matplotlib', obj, 'style')
        self.assertEqual(custom_obj_lookup.kwargs, expected_custom_obj)


@attr(optional=1) # Requires matplotlib
class TestStoreInheritance(ComparisonTestCase):
    """
    Tests to prevent regression after fix in 71c1f3a that resolves
    issue #43
    """

    def setUp(self):
        self.store_copy = OptionTree(sorted(Store.options().items()),
                                     groups=['style', 'plot', 'norm'])
        self.backend = 'matplotlib'
        Store.current_backend = self.backend
        Store.options(val=OptionTree(groups=['plot', 'style']))

        options = Store.options()

        self.default_plot = dict(plot1='plot1', plot2='plot2')
        options.Histogram = Options('plot', **self.default_plot)

        self.default_style = dict(style1='style1', style2='style2')
        options.Histogram = Options('style', **self.default_style)

        data = [np.random.normal() for i in range(10000)]
        frequencies, edges = np.histogram(data, 20)
        self.hist = Histogram(frequencies, edges)
        super(TestStoreInheritance, self).setUp()


    def lookup_options(self, obj, group):
        return Store.lookup_options(self.backend, obj, group)

    def tearDown(self):
        Store.options(val=self.store_copy)
        Store._custom_options = {k:{} for k in Store._custom_options.keys()}
        super(TestStoreInheritance, self).tearDown()

    def test_original_style_options(self):
        self.assertEqual(self.lookup_options(self.hist, 'style').options,
                         self.default_style)

    def test_original_plot_options(self):
        self.assertEqual(self.lookup_options(self.hist, 'plot').options,
                         self.default_plot)

    def test_plot_inheritance_addition(self):
        "Adding an element"
        hist2 = self.hist(plot={'plot3':'plot3'})
        self.assertEqual(self.lookup_options(hist2, 'plot').options,
                         dict(plot1='plot1', plot2='plot2', plot3='plot3'))
        # Check style works as expected
        self.assertEqual(self.lookup_options(hist2, 'style').options, self.default_style)

    def test_plot_inheritance_override(self):
        "Overriding an element"
        hist2 = self.hist(plot={'plot1':'plot_child'})
        self.assertEqual(self.lookup_options(hist2, 'plot').options,
                         dict(plot1='plot_child', plot2='plot2'))
        # Check style works as expected
        self.assertEqual(self.lookup_options(hist2, 'style').options, self.default_style)

    def test_style_inheritance_addition(self):
        "Adding an element"
        hist2 = self.hist(style={'style3':'style3'})
        self.assertEqual(self.lookup_options(hist2, 'style').options,
                         dict(style1='style1', style2='style2', style3='style3'))
        # Check plot options works as expected
        self.assertEqual(self.lookup_options(hist2, 'plot').options, self.default_plot)

    def test_style_inheritance_override(self):
        "Overriding an element"
        hist2 = self.hist(style={'style1':'style_child'})
        self.assertEqual(self.lookup_options(hist2, 'style').options,
                         dict(style1='style_child', style2='style2'))
        # Check plot options works as expected
        self.assertEqual(self.lookup_options(hist2, 'plot').options, self.default_plot)

    def test_style_transfer(self):
        hist = self.hist.opts(style={'style1':'style_child'})
        hist2 = self.hist.opts()
        opts = Store.lookup_options('matplotlib', hist2, 'style').kwargs
        self.assertEqual(opts, {'style1': 'style1', 'style2': 'style2'})
        Store.transfer_options(hist, hist2, 'matplotlib')
        opts = Store.lookup_options('matplotlib', hist2, 'style').kwargs
        self.assertEqual(opts, {'style1': 'style_child', 'style2': 'style2'})


@attr(optional=1) # Needs matplotlib
class TestOptionTreeFind(ComparisonTestCase):

    def setUp(self):
        options = OptionTree(groups=['group'])
        self.opts1 = Options('group', kw1='value1')
        self.opts2 = Options('group', kw2='value2')
        self.opts3 = Options('group', kw3='value3')
        self.opts4 = Options('group', kw4='value4')
        self.opts5 = Options('group', kw5='value5')
        self.opts6 = Options('group', kw6='value6')

        options.MyType = self.opts1
        options.XType = self.opts2
        options.MyType.Foo = self.opts3
        options.MyType.Bar = self.opts4
        options.XType.Foo = self.opts5
        options.XType.Bar = self.opts6

        self.options = options
        self.original_options = Store.options()
        Store.options(val = OptionTree(groups=['group']))


    def tearDown(self):
        Store.options(val=self.original_options)
        Store._custom_options = {k:{} for k in Store._custom_options.keys()}

    def test_optiontree_find1(self):
        self.assertEqual(self.options.find('MyType').options('group').options,
                         dict(kw1='value1'))

    def test_optiontree_find2(self):
        self.assertEqual(self.options.find('XType').options('group').options,
                         dict(kw2='value2'))

    def test_optiontree_find3(self):
        self.assertEqual(self.options.find('MyType.Foo').options('group').options,
                         dict(kw1='value1', kw3='value3'))

    def test_optiontree_find4(self):
        self.assertEqual(self.options.find('MyType.Bar').options('group').options,
                         dict(kw1='value1', kw4='value4'))

    def test_optiontree_find5(self):
        self.assertEqual(self.options.find('XType.Foo').options('group').options,
                         dict(kw2='value2', kw5='value5'))

    def test_optiontree_find6(self):
        self.assertEqual(self.options.find('XType.Bar').options('group').options,
                         dict(kw2='value2', kw6='value6'))

    def test_optiontree_find_mismatch1(self):
        self.assertEqual(self.options.find('MyType.Baz').options('group').options,
                         dict(kw1='value1'))

    def test_optiontree_find_mismatch2(self):
        self.assertEqual(self.options.find('XType.Baz').options('group').options,
                         dict(kw2='value2'))

    def test_optiontree_find_mismatch3(self):
        self.assertEqual(self.options.find('Baz').options('group').options, dict())

    def test_optiontree_find_mismatch4(self):
        self.assertEqual(self.options.find('Baz.Baz').options('group').options, dict())



class TestCrossBackendOptions(ComparisonTestCase):
    """
    Test the style system can style a single object across backends.
    """

    def setUp(self):

        if 'bokeh' not in Store.renderers:
            raise SkipTest("Cross background tests assumes bokeh is available.")
        self.store_mpl = OptionTree(sorted(Store.options(backend='matplotlib').items()),
                                    groups=['style', 'plot', 'norm'])
        self.store_bokeh = OptionTree(sorted(Store.options(backend='bokeh').items()),
                                    groups=['style', 'plot', 'norm'])
        self.clear_options()
        super(TestCrossBackendOptions, self).setUp()


    def clear_options(self):
        # Clear global options..
        Store.options(val=OptionTree(groups=['plot', 'style']), backend='matplotlib')
        Store.options(val=OptionTree(groups=['plot', 'style']), backend='bokeh')
        # ... and custom options
        Store.custom_options({}, backend='matplotlib')
        Store.custom_options({}, backend='bokeh')


    def tearDown(self):
        Store.options(val=self.store_mpl, backend='matplotlib')
        Store.options(val=self.store_bokeh, backend='bokeh')
        Store.current_backend = 'matplotlib'
        Store._custom_options = {k:{} for k in Store._custom_options.keys()}
        super(TestCrossBackendOptions, self).tearDown()


    def test_mpl_bokeh_mpl(self):
        img = Image(np.random.rand(10,10))
        # Use blue in matplotlib
        Store.current_backend = 'matplotlib'
        StoreOptions.set_options(img, style={'Image':{'cmap':'Blues'}})
        mpl_opts = Store.lookup_options('matplotlib', img, 'style').options
        self.assertEqual(mpl_opts, {'cmap':'Blues'})
        # Use purple in bokeh
        Store.current_backend = 'bokeh'
        StoreOptions.set_options(img, style={'Image':{'cmap':'Purple'}})
        bokeh_opts = Store.lookup_options('bokeh', img, 'style').options
        self.assertEqual(bokeh_opts, {'cmap':'Purple'})
        # Check it is still blue in matplotlib...
        Store.current_backend = 'matplotlib'
        mpl_opts = Store.lookup_options('matplotlib', img, 'style').options
        self.assertEqual(mpl_opts, {'cmap':'Blues'})
        # And purple in bokeh..
        Store.current_backend = 'bokeh'
        bokeh_opts = Store.lookup_options('bokeh', img, 'style').options
        self.assertEqual(bokeh_opts, {'cmap':'Purple'})
        return img


    def test_mpl_bokeh_offset_mpl(self):
        img = Image(np.random.rand(10,10))
        # Use blue in matplotlib
        Store.current_backend = 'matplotlib'
        StoreOptions.set_options(img, style={'Image':{'cmap':'Blues'}})
        mpl_opts = Store.lookup_options('matplotlib', img, 'style').options
        self.assertEqual(mpl_opts, {'cmap':'Blues'})
        # Switch to bokeh and style a random object...
        Store.current_backend = 'bokeh'
        img2 = Image(np.random.rand(10,10))
        StoreOptions.set_options(img2, style={'Image':{'cmap':'Reds'}})
        img2_opts = Store.lookup_options('bokeh', img2, 'style').options
        self.assertEqual(img2_opts, {'cmap':'Reds'})
        # Use purple in bokeh on the object...
        StoreOptions.set_options(img, style={'Image':{'cmap':'Purple'}})
        bokeh_opts = Store.lookup_options('bokeh', img, 'style').options
        self.assertEqual(bokeh_opts, {'cmap':'Purple'})
        # Check it is still blue in matplotlib...
        Store.current_backend = 'matplotlib'
        mpl_opts = Store.lookup_options('matplotlib', img, 'style').options
        self.assertEqual(mpl_opts, {'cmap':'Blues'})
        # And purple in bokeh..
        Store.current_backend = 'bokeh'
        bokeh_opts = Store.lookup_options('bokeh', img, 'style').options
        self.assertEqual(bokeh_opts, {'cmap':'Purple'})
        return img


class TestCrossBackendOptionPickling(TestCrossBackendOptions):

    cleanup = ['test_raw_pickle.pkl', 'test_pickle_mpl_bokeh.pkl']

    def tearDown(self):
        super(TestCrossBackendOptionPickling, self).tearDown()
        for f in self.cleanup:
            try:
                os.remove(f)
            except:
                pass

    def test_raw_pickle(self):
        """
        Test usual pickle saving and loading (no style information preserved)
        """
        fname= 'test_raw_pickle.pkl'
        raw = super(TestCrossBackendOptionPickling, self).test_mpl_bokeh_mpl()
        pickle.dump(raw, open(fname,'wb'))
        self.clear_options()
        img = pickle.load(open(fname,'rb'))
        # Data should match
        self.assertEqual(raw, img)
        # But the styles will be lost without using Store.load/Store.dump
        pickle.current_backend = 'matplotlib'
        mpl_opts = Store.lookup_options('matplotlib', img, 'style').options
        self.assertEqual(mpl_opts, {})
        # ... across all backends
        Store.current_backend = 'bokeh'
        bokeh_opts = Store.lookup_options('bokeh', img, 'style').options
        self.assertEqual(bokeh_opts, {})

    def test_pickle_mpl_bokeh(self):
        """
        Test pickle saving and loading with Store (style information preserved)
        """
        fname = 'test_pickle_mpl_bokeh.pkl'
        raw = super(TestCrossBackendOptionPickling, self).test_mpl_bokeh_mpl()
        Store.dump(raw, open(fname,'wb'))
        self.clear_options()
        img = Store.load(open(fname,'rb'))
        # Data should match
        self.assertEqual(raw, img)
        # Check it is still blue in matplotlib...
        Store.current_backend = 'matplotlib'
        mpl_opts = Store.lookup_options('matplotlib', img, 'style').options
        self.assertEqual(mpl_opts, {'cmap':'Blues'})
        # And purple in bokeh..
        Store.current_backend = 'bokeh'
        bokeh_opts = Store.lookup_options('bokeh', img, 'style').options
        self.assertEqual(bokeh_opts, {'cmap':'Purple'})


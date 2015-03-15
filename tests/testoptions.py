from holoviews import Store
from holoviews.core.options import OptionError, Cycle, Options, OptionTree
from holoviews.element.comparison import ComparisonTestCase


class TestOptions(ComparisonTestCase):

    def test_options_init(self):
        Options('test')

    def test_options_valid_keywords1(self):
        Options('test', allowed_keywords=['kw1'], kw1='value')

    def test_options_valid_keywords2(self):
        Options('test', allowed_keywords=['kw1', 'kw2'], kw1='value')

    def test_options_valid_keywords3(self):
        Options('test', allowed_keywords=['kw1', 'kw2'], kw1='value', kw2='value')

    def test_options_any_keywords3(self):
        Options('test', kw1='value', kw2='value')

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


    def test_cycle_slice(self):
        cycle1 = Cycle(values=['a', 'b', 'c'])[2]
        cycle2 = Cycle(values=[1, 2, 3])

        opts = Options('test', one=cycle1, two=cycle2)
        self.assertEqual(opts[0], {'one': 'a', 'two': 1})
        self.assertEqual(opts[1], {'one': 'b', 'two': 2})
        self.assertEqual(opts[2], {'one': 'a', 'two': 1})
        self.assertEqual(opts[3], {'one': 'b', 'two': 2})


    def test_options_property_disabled(self):
        cycle1 = Cycle(values=['a', 'b', 'c'])
        opts = Options('test', one=cycle1)
        try:
            opts.options
        except Exception as e:
            self.assertEqual(str(e), "The options property may only be used with non-cyclic Options.")



class TestOptionTree(ComparisonTestCase):

    def test_optiontree_init(self):
        OptionTree(groups={'group1':  Options(),
                           'group2': Options()})

    def test_optiontree_setter_getter(self):
        options = OptionTree(groups={'group1':  Options(),
                                     'group2': Options()})
        opts = Options('group1', kw1='value')
        options.MyType = opts
        self.assertEqual(options.MyType['group1'], opts)
        self.assertEqual(options.MyType['group1'].options, {'kw1':'value'})

    def test_optiontree_dict_setter_getter(self):
        options = OptionTree(groups={'group1':  Options(),
                                     'group2': Options()})

        opts1 = Options(kw1='value1')
        opts2 = Options(kw2='value2')
        options.MyType = {'group1':opts1, 'group2':opts2}

        self.assertEqual(options.MyType['group1'], opts1)
        self.assertEqual(options.MyType['group1'].options, {'kw1':'value1'})

        self.assertEqual(options.MyType['group2'], opts2)
        self.assertEqual(options.MyType['group2'].options, {'kw2':'value2'})

    def test_optiontree_inheritance(self):
        options = OptionTree(groups={'group1':  Options(),
                                     'group2': Options()})

        opts1 = Options(kw1='value1')
        opts2 = Options(kw2='value2')
        options.MyType = {'group1':opts1, 'group2':opts2}

        opts3 = Options(kw3='value3')
        opts4 = Options(kw4='value4')
        options.MyType.Child = {'group1':opts3, 'group2':opts4}

        self.assertEqual(options.MyType.Child['group1'].options,
                         {'kw1':'value1', 'kw3':'value3'})

        self.assertEqual(options.MyType.Child['group2'].options,
                         {'kw2':'value2', 'kw4':'value4'})


class TestOptionTreeFind(ComparisonTestCase):

    def setUp(self):
        options = OptionTree(groups={'group':  Options()})
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
        self.original_options = Store.options
        Store.options = OptionTree(groups={'group':  Options()})


    def tearDown(self):
        Store.options = self.original_options


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



if __name__ == "__main__":
    import sys
    import nose
    nose.runmodule(argv=[sys.argv[0], "--logging-level", "ERROR"])

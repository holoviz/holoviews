from collections import namedtuple

from .utils import ViewTestCase

from dataviews.options import OptionsGroup, Options
from dataviews.options import PlotOpts, StyleOpts, ChannelOpts, Cycle
from dataviews.views import View



class TestOpts(ViewTestCase):

    def test_styleopts_init(self):
        opt = StyleOpts(key1='key1', key2='key2')
        self.assertEqual(opt.items, dict(key1='key1', key2='key2'))

    def test_plotopts_init(self):
        opt = PlotOpts(opt1='opt1', opt2='opt2')
        self.assertEqual(opt.items, dict(opt1='opt1', opt2='opt2'))

    def test_channelopts_init(self):
        opt = ChannelOpts('RGBA', 'view1 * view2', opt1='opt1', opt2='opt2')
        self.assertEqual(opt.mode, 'RGBA')
        self.assertEqual(opt.pattern, 'view1 * view2')
        self.assertEqual(opt.items, dict(opt1='opt1', opt2='opt2'))


    def test_styleopts_methods(self):
        kwargs = dict(key2='key2', key1='key1')
        opt = StyleOpts(**kwargs)
        self.assertEqual(opt.keys(), ['key1', 'key2'])
        for i in range(16):
            self.assertEqual(opt[i], kwargs)

    def test_styleopts_cycles(self):
        cycle1 = Cycle(['a', 'b', 'c'])
        cycle2 = Cycle([1, 2, 3])
        opts = StyleOpts(one=cycle1, two=cycle2)
        self.assertEqual(opts[0], {'one': 'a', 'two': 1})
        self.assertEqual(opts[1], {'one': 'b', 'two': 2})
        self.assertEqual(opts[2], {'one': 'c', 'two': 3})
        self.assertEqual(opts[3], {'one': 'a', 'two': 1})
        self.assertEqual(opts[4], {'one': 'b', 'two': 2})
        self.assertEqual(opts[5], {'one': 'c', 'two': 3})

    def test_styleopts_opts_property_noncyclic(self):
        kwargs = dict(key2='key2', key1='key1')
        opt = StyleOpts(**kwargs)
        self.assertEqual(opt.opts, kwargs)

    def test_styleopts_opts_property_cyclic(self):
        cycle1 = Cycle(['a', 'b', 'c'])
        cycle2 = Cycle([1, 2, 3])
        opts = StyleOpts(one=cycle1, two=cycle2)
        try:
            self.assertEqual(opts.opts, {'one': 'a', 'two': 1})
            raise AssertionError("Opts property only applicable without cycles")
        except Exception as e:
            assert str(e) == "The opts property may only be used with non-cyclic styles"

    def test_styleopts_cycle_mismatch(self):
        cycle1 = Cycle(['a', 'b',])
        cycle2 = Cycle([1, 2, 3])
        try:
            StyleOpts(one=cycle1, two=cycle2)
            raise AssertionError("Cycle length mismatch not detected")
        except Exception as e:
            assert str(e) == 'Cycle objects supplied with different lengths'


    def test_styleopts_inherit(self):
        kwargs = dict(key2='key2', key1='key1')
        opt = StyleOpts(**kwargs)
        self.assertEqual(opt(key3='key3').opts,
                         dict(kwargs, key3='key3'))



class TestOptions(ViewTestCase):

    def setUp(self):
        self.s1_kws = dict(key1='style1_key1', key2='style1_key2')
        self.s2_kws = dict(key3='style2_key3', key4='style2_key4')
        self.s3_kws = dict(key5='style3_key5', key6='style3_key6')
        self.style1 = StyleOpts(**self.s1_kws)
        self.style2 = StyleOpts(**self.s2_kws)
        self.style3 = StyleOpts(**self.s3_kws)
        self.custom = StyleOpts(key5='custom_key5', key6='custom_key6')

    def test_style_options_init(self):
        options = Options('style', StyleOpts)

    def test_plot_options_init(self):
        options = Options('plotting', PlotOpts)

    def test_channel_options_init(self):
        options = Options('channels', ChannelOpts)


    def test_channel_options_init_invalid(self):
        try:
            Options('invalid', None)
            raise AssertionError("None is not an Opts object.")
        except TypeError as e:
            assert str(e) == "issubclass() arg 1 must be a class"

    def test_options_access(self):
        options = Options('style', StyleOpts)
        try:
            options.set('style1', self.style1)
            raise AssertionError("Options should only be set through OptionGroups")
        except Exception as e:
            assert str(e) == "OptionMaps should be set via OptionGroup"

    def test_option_keys(self):
        options = Options('style', StyleOpts)
        options._settable = True
        options.set('style1', self.style1)
        options.set('style2', self.style2)
        options.set('Custom', self.custom)
        self.assertEqual(set(options.keys()), set(['style1','style2', 'Custom']))
        self.assertEqual(set(options.options()), set(['style1','style2']))

    def test_option_contains(self):
        options = Options('style', StyleOpts)
        options._settable = True
        options.set('style1', self.style1)
        options.set('style2', self.style2)
        options.set('Custom', self.custom)
        self.assertEqual('style1' in options, True)
        self.assertEqual('style4' in options, False)

    def test_option_access(self):
        options = Options('style', StyleOpts)
        options._settable = True
        options.set('style1', self.style1)
        options.set('style2', self.style2)
        options.set('Custom', self.custom)

        self.assertEqual(options['style1'], self.style1)
        self.assertEqual(options['invalid'], StyleOpts())
        self.assertEqual(options.style1, self.style1)
        try:
            self.assertEqual(options.invalid, StyleOpts())
            raise AssertionError("AttributeError should be raised for invalid key")
        except AttributeError as e:
            assert str(e) == 'invalid'

    def test_option_fuzzy_match_two_levels(self):
        options = Options('style', StyleOpts)
        options._settable = True
        options.set('style', self.style1)
        options.set('specific_style', self.style2)
        self.assertEqual(options('nomatch'), StyleOpts())
        self.assertEqual(options('style').opts, self.s1_kws)
        self.assertEqual(options('specific_style').opts,
                         dict(self.s1_kws, **self.s2_kws))
        self.assertEqual(options('nonmatching_style').opts, self.s1_kws)

    def test_option_fuzzy_match_three_levels(self):
        options = Options('style', StyleOpts)
        options._settable = True
        options.set('style', self.style1)
        options.set('specific_style', self.style2)
        options.set('very_specific_style', self.style3)
        self.assertEqual(options('nomatch'), StyleOpts())
        self.assertEqual(options('fairly_specific_style').opts,
                         dict(self.s1_kws, **self.s2_kws))
        self.assertEqual(options('very_specific_style').opts,
                         dict(list(self.s1_kws.items())
                            + list(self.s2_kws.items())
                            + list(self.s3_kws.items())))

    def test_option_fuzzy_match_object(self):
        styled = namedtuple('Test', 'style')
        style_obj = styled(style='specific_style')

        options = Options('style', StyleOpts)
        options._settable = True
        options.set('style', self.style1)
        options.set('specific_style', self.style2)
        self.assertEqual(options(style_obj).opts,
                         dict(self.s1_kws, **self.s2_kws))

class TestOptionGroup(ViewTestCase):

    def setUp(self):
        self.s1 = StyleOpts(key1='key1', key2='key2')
        self.s2 = StyleOpts(key3='key3', key4='key4')
        self.s3 = StyleOpts(key5='key5', key6='key6')

        self.p1 = PlotOpts(opt1='opt1', opt2='opt2')
        self.p2 = PlotOpts(opt3='opt3', opt4='opt4')

    def test_option_group_init(self):
        OptionsGroup([Options('plotting', PlotOpts),
                      Options('style', StyleOpts)])

    def test_option_group_access(self):
        optgroup = OptionsGroup([Options('plotting', PlotOpts),
                                 Options('style', StyleOpts)])
        # StyleOpts attribute setting
        optgroup.name1 = self.s1
        optgroup.name2 = self.s2
        # PlotOpts attribute setting
        optgroup.name1 = self.p1
        optgroup.name3 = self.p2
        # Dictionary style access
        for (opt1,opt2) in zip(optgroup['name1'], [self.p1, self.s1]):
            self.assertEqual(opt1, opt2)
        # Attribute style access
        for (opt1,opt2) in zip(optgroup.name2, [PlotOpts(), self.s2]):
            self.assertEqual(opt1, opt2)
        for (opt1,opt2) in zip(optgroup.name3, [self.p2, StyleOpts()]):
            self.assertEqual(opt1, opt2)


    def test_option_group_keys(self):
        optgroup = OptionsGroup([Options('plotting', PlotOpts),
                                 Options('style', StyleOpts)])
        optgroup.name1 = self.s1
        optgroup.name2 = self.s2
        optgroup.name1 = self.p1
        optgroup.name3 = self.p2
        optgroup.Custom_key = self.p2

        self.assertEqual(optgroup.options(), ['name1', 'name2', 'name3'])
        self.assertEqual(optgroup.keys(), ['Custom_key', 'name1', 'name2', 'name3'])


    def test_option_group_fuzzy_match_keys(self):
        optgroup = OptionsGroup([Options('plotting', PlotOpts),
                                 Options('style', StyleOpts)])
        optgroup.unrelated = self.s1
        self.assertEqual(optgroup.fuzzy_match_keys('name'), [])
        optgroup.Name = self.s1
        optgroup.SpecificName = self.s2
        self.assertEqual(optgroup.fuzzy_match_keys('SpecificName'),
                         ['SpecificName', 'Name'])

        self.assertEqual(optgroup.fuzzy_match_keys('SpecificWrongName'), ['Name'])

        self.assertEqual(optgroup.fuzzy_match_keys('VerySpecificName'),
                         ['SpecificName', 'Name'])

    def test_option_fuzzy_match_view(self):
        optgroup = OptionsGroup([Options('plotting', PlotOpts),
                                 Options('style', StyleOpts)])
        View.options = optgroup
        v = View(None, label='Test')
        optgroup.Test_View = self.s1
        self.assertEqual(v.label, 'Test')
        self.assertEqual(v.style, 'Test_View')
        self.assertEqual(v.options.fuzzy_match_keys('Test_View'), ['Test_View'])



if __name__ == "__main__":
    import sys
    import nose
    nose.runmodule(argv=[sys.argv[0], "--logging-level", "ERROR"])

from unittest import SkipTest

from holoviews.core.dimension import Dimensioned
from holoviews.core.options import Store, Keywords, Options, OptionTree
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase

class TestObj(Dimensioned):
    pass


class CustomBackendTestCase(LoggingComparisonTestCase):
    """
    Registers fake backends with the Store to test options on.
    """

    def setUp(self):
        self.current_backend = Store.current_backend
        self.register_custom(TestObj, 'backend_1', ['plot_custom1'])
        self.register_custom(TestObj, 'backend_2', ['plot_custom2'])
        Store.current_backend = 'backend_1'

    def tearDown(self):
        Store._options.pop('backend_1')
        Store._options.pop('backend_2')
        Store.current_backend = self.current_backend

    @classmethod
    def register_custom(cls, objtype, backend, custom_plot=[], custom_style=[]):
        groups = ['style', 'plot', 'norm']
        if backend not in Store._options:
            Store._options[backend] = OptionTree([], groups=groups)
            Store._custom_options[backend] = {}
        name = objtype.__name__
        groups = ['style', 'plot', 'norm']
        style_opts = Keywords(['style_opt1', 'style_opt2']+custom_style, name)
        plot_opts = Keywords(['plot_opt1', 'plot_opt2']+custom_plot, name)
        opt_groups = {'plot': Options(allowed_keywords=plot_opts),
                      'style': Options(allowed_keywords=style_opts)}
        Store._options[backend][name] = opt_groups



class TestDimensioned_options(CustomBackendTestCase):

    def test_apply_options_current_backend_style(self):
        obj = TestObj([]).options(style_opt1='A')
        opts = Store.lookup_options('backend_1', obj, 'style')
        assert opts.options == {'style_opt1': 'A'}

    def test_apply_options_current_backend_style_invalid(self):
        err = ("Unexpected option 'style_opt3' for TestObj types "
               "across all extensions. Similar options for current "
               "extension \('backend_1'\) are: \['style_opt1', 'style_opt2'\]\.")
        with self.assertRaisesRegexp(ValueError, err):
            TestObj([]).options(style_opt3='A')

    def test_apply_options_current_backend_style_invalid_no_match(self):
        err = ("Unexpected option 'zxy' for TestObj types across all extensions\. "
               "No similar options found\.")
        with self.assertRaisesRegexp(ValueError, err):
            TestObj([]).options(zxy='A')

    def test_apply_options_explicit_backend_style_invalid(self):
        err = ("Unexpected option 'style_opt3' for TestObj types "
               "across all extensions. Similar options for current "
               "extension \('backend_2'\) are: \['style_opt1', 'style_opt2'\]\.")
        with self.assertRaisesRegexp(ValueError, err):
            TestObj([]).options(style_opt3='A', backend='backend_2')

    def test_apply_options_explicit_backend_style_invalid_no_match(self):
        err = ("Unexpected option 'zxy' for TestObj types when using the "
               "'backend_2' extension. No similar options founds\.")
        
        with self.assertRaisesRegexp(ValueError, err):
            TestObj([]).options(zxy='A', backend='backend_2')

    def test_apply_options_current_backend_style_invalid_cross_backend_match(self):
        TestObj([]).options(plot_custom2='A')
        substr = ("Option 'plot_custom2' for TestObj type not valid for "
                  "selected backend ('backend_1'). Option only applies to "
                  "following backends: ['backend_2']")
        self.log_handler.assertEndsWith('WARNING', substr)

    def test_apply_options_explicit_backend_style_invalid(self):
        err = ("Unexpected option 'style_opt3' for TestObj types when "
               "using the 'backend_2' extension. Similar options are: "
               "\['style_opt1', 'style_opt2'\]\.")
        with self.assertRaisesRegexp(ValueError, err):
            TestObj([]).options(style_opt3='A', backend='backend_2')

    def test_apply_options_current_backend_style_multiple(self):
        obj = TestObj([]).options(style_opt1='A', style_opt2='B')
        opts = Store.lookup_options('backend_1', obj, 'style')
        assert opts.options == {'style_opt1': 'A', 'style_opt2': 'B'}

    def test_apply_options_current_backend_plot(self):
        obj = TestObj([]).options(plot_opt1='A')
        opts = Store.lookup_options('backend_1', obj, 'plot')
        assert opts.options == {'plot_opt1': 'A'}

    def test_apply_options_current_backend_plot_multiple(self):
        obj = TestObj([]).options(plot_opt1='A', plot_opt2='B')
        opts = Store.lookup_options('backend_1', obj, 'plot')
        assert opts.options == {'plot_opt1': 'A', 'plot_opt2': 'B'}

    def test_apply_options_current_backend_plot_and_style(self):
        obj = TestObj([]).options(style_opt1='A', plot_opt1='B')
        plot_opts = Store.lookup_options('backend_1', obj, 'plot')
        assert plot_opts.options == {'plot_opt1': 'B'}
        style_opts = Store.lookup_options('backend_1', obj, 'style')
        assert style_opts.options == {'style_opt1': 'A'}

    def test_apply_options_explicit_backend_style(self):
        obj = TestObj([]).options(style_opt1='A', backend='backend_2')
        opts = Store.lookup_options('backend_2', obj, 'style')
        assert opts.options == {'style_opt1': 'A'}

    def test_apply_options_explicit_backend_style_multiple(self):
        obj = TestObj([]).options(style_opt1='A', style_opt2='B', backend='backend_2')
        opts = Store.lookup_options('backend_2', obj, 'style')
        assert opts.options == {'style_opt1': 'A', 'style_opt2': 'B'}

    def test_apply_options_explicit_backend_plot(self):
        obj = TestObj([]).options(plot_opt1='A', backend='backend_2')
        opts = Store.lookup_options('backend_2', obj, 'plot')
        assert opts.options == {'plot_opt1': 'A'}

    def test_apply_options_explicit_backend_plot_multiple(self):
        obj = TestObj([]).options(plot_opt1='A', plot_opt2='B', backend='backend_2')
        opts = Store.lookup_options('backend_2', obj, 'plot')
        assert opts.options == {'plot_opt1': 'A', 'plot_opt2': 'B'}

    def test_apply_options_explicit_backend_plot_and_style(self):
        obj = TestObj([]).options(style_opt1='A', plot_opt1='B', backend='backend_2')
        plot_opts = Store.lookup_options('backend_2', obj, 'plot')
        assert plot_opts.options == {'plot_opt1': 'B'}
        style_opts = Store.lookup_options('backend_2', obj, 'style')
        assert style_opts.options == {'style_opt1': 'A'}

    def test_apply_options_not_cloned(self):
        obj1 = TestObj([])
        obj2 = obj1.options(style_opt1='A', clone=False)
        opts = Store.lookup_options('backend_1', obj1, 'style')
        assert opts.options == {'style_opt1': 'A'}
        assert obj1 is obj2
    
    def test_apply_options_cloned(self):
        obj1 = TestObj([])
        obj2 = obj1.options(style_opt1='A')
        opts = Store.lookup_options('backend_1', obj2, 'style')
        assert opts.options == {'style_opt1': 'A'}
        assert obj1 is not obj2

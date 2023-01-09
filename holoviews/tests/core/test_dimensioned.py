import gc

from holoviews.core.spaces import HoloMap
from holoviews.core.element import Element
from holoviews.core.options import Store, Keywords, Options, OptionTree
from ..utils import LoggingComparisonTestCase


class ExampleElement(Element):
    pass


class MockRenderer:

    def __init__(self, backend):
        self.backend = backend


class CustomBackendTestCase(LoggingComparisonTestCase):
    """
    Registers fake backends with the Store to test options on.
    """

    def setUp(self):
        super().setUp()
        self.current_backend = Store.current_backend
        self.register_custom(ExampleElement, 'backend_1', ['plot_custom1'])
        self.register_custom(ExampleElement, 'backend_2', ['plot_custom2'])
        Store.set_current_backend('backend_1')

    def tearDown(self):
        super().tearDown()
        Store._weakrefs = {}
        Store._options.pop('backend_1')
        Store._options.pop('backend_2')
        Store._custom_options.pop('backend_1')
        Store._custom_options.pop('backend_2')
        Store.set_current_backend(self.current_backend)
        Store.renderers.pop('backend_1')
        Store.renderers.pop('backend_2')

    @classmethod
    def register_custom(cls, objtype, backend, custom_plot=[], custom_style=[]):
        groups = Options._option_groups
        if backend not in Store._options:
            Store._options[backend] = OptionTree([], groups=groups)
            Store._custom_options[backend] = {}
        name = objtype.__name__
        style_opts = Keywords(['style_opt1', 'style_opt2']+custom_style, name)
        plot_opts = Keywords(['plot_opt1', 'plot_opt2']+custom_plot, name)
        opt_groups = {'plot': Options(allowed_keywords=plot_opts),
                      'style': Options(allowed_keywords=style_opts),
                      'output': Options(allowed_keywords=['backend'])}
        Store._options[backend][name] = opt_groups
        Store.renderers[backend] = MockRenderer(backend)



class TestDimensioned_options(CustomBackendTestCase):

    def test_apply_options_current_backend_style(self):
        obj = ExampleElement([]).options(style_opt1='A')
        opts = Store.lookup_options('backend_1', obj, 'style')
        assert opts.options == {'style_opt1': 'A'}

    def test_apply_options_current_backend_style_invalid(self):
        err = ("Unexpected option 'style_opt3' for ExampleElement type "
               "across all extensions. Similar options for current "
               r"extension \('backend_1'\) are: \['style_opt1', 'style_opt2'\]\.")
        with self.assertRaisesRegex(ValueError, err):
            ExampleElement([]).options(style_opt3='A')

    def test_apply_options_current_backend_style_invalid_no_match(self):
        err = (r"Unexpected option 'zxy' for ExampleElement type across all extensions\. "
               r"No similar options found\.")
        with self.assertRaisesRegex(ValueError, err):
            ExampleElement([]).options(zxy='A')

    def test_apply_options_explicit_backend_style_invalid_cross_backend(self):
        err = ("Unexpected option 'style_opt3' for ExampleElement type when "
               "using the 'backend_2' extension. Similar options are: "
               r"\['style_opt1', 'style_opt2'\]\.")
        with self.assertRaisesRegex(ValueError, err):
            ExampleElement([]).options(style_opt3='A', backend='backend_2')

    def test_apply_options_explicit_backend_style_invalid_no_match(self):
        err = ("Unexpected option 'zxy' for ExampleElement type when using the "
               r"'backend_2' extension. No similar options found\.")
        with self.assertRaisesRegex(ValueError, err):
            ExampleElement([]).options(zxy='A', backend='backend_2')

    def test_apply_options_current_backend_style_invalid_cross_backend_match(self):
        ExampleElement([]).options(plot_custom2='A')
        substr = ("Option 'plot_custom2' for ExampleElement type not valid for "
                  "selected backend ('backend_1'). Option only applies to "
                  "following backends: ['backend_2']")
        self.log_handler.assertEndsWith('WARNING', substr)

    def test_apply_options_explicit_backend_style_invalid(self):
        err = ("Unexpected option 'style_opt3' for ExampleElement type when "
               "using the 'backend_2' extension. Similar options are: "
               r"\['style_opt1', 'style_opt2'\]\.")
        with self.assertRaisesRegex(ValueError, err):
            ExampleElement([]).options(style_opt3='A', backend='backend_2')

    def test_apply_options_current_backend_style_multiple(self):
        obj = ExampleElement([]).options(style_opt1='A', style_opt2='B')
        opts = Store.lookup_options('backend_1', obj, 'style')
        assert opts.options == {'style_opt1': 'A', 'style_opt2': 'B'}

    def test_apply_options_current_backend_plot(self):
        obj = ExampleElement([]).options(plot_opt1='A')
        opts = Store.lookup_options('backend_1', obj, 'plot')
        assert opts.options == {'plot_opt1': 'A'}

    def test_apply_options_current_backend_plot_multiple(self):
        obj = ExampleElement([]).options(plot_opt1='A', plot_opt2='B')
        opts = Store.lookup_options('backend_1', obj, 'plot')
        assert opts.options == {'plot_opt1': 'A', 'plot_opt2': 'B'}

    def test_apply_options_current_backend_plot_and_style(self):
        obj = ExampleElement([]).options(style_opt1='A', plot_opt1='B')
        plot_opts = Store.lookup_options('backend_1', obj, 'plot')
        assert plot_opts.options == {'plot_opt1': 'B'}
        style_opts = Store.lookup_options('backend_1', obj, 'style')
        assert style_opts.options == {'style_opt1': 'A'}

    def test_apply_options_explicit_backend_style(self):
        obj = ExampleElement([]).options(style_opt1='A', backend='backend_2')
        opts = Store.lookup_options('backend_2', obj, 'style')
        assert opts.options == {'style_opt1': 'A'}

    def test_apply_options_explicit_backend_style_multiple(self):
        obj = ExampleElement([]).options(style_opt1='A', style_opt2='B', backend='backend_2')
        opts = Store.lookup_options('backend_2', obj, 'style')
        assert opts.options == {'style_opt1': 'A', 'style_opt2': 'B'}

    def test_apply_options_explicit_backend_plot(self):
        obj = ExampleElement([]).options(plot_opt1='A', backend='backend_2')
        opts = Store.lookup_options('backend_2', obj, 'plot')
        assert opts.options == {'plot_opt1': 'A'}

    def test_apply_options_explicit_backend_plot_multiple(self):
        obj = ExampleElement([]).options(plot_opt1='A', plot_opt2='B', backend='backend_2')
        opts = Store.lookup_options('backend_2', obj, 'plot')
        assert opts.options == {'plot_opt1': 'A', 'plot_opt2': 'B'}

    def test_apply_options_explicit_backend_plot_and_style(self):
        obj = ExampleElement([]).options(style_opt1='A', plot_opt1='B', backend='backend_2')
        plot_opts = Store.lookup_options('backend_2', obj, 'plot')
        assert plot_opts.options == {'plot_opt1': 'B'}
        style_opts = Store.lookup_options('backend_2', obj, 'style')
        assert style_opts.options == {'style_opt1': 'A'}

    def test_apply_options_not_cloned(self):
        obj1 = ExampleElement([])
        obj2 = obj1.options(style_opt1='A', clone=False)
        opts = Store.lookup_options('backend_1', obj1, 'style')
        assert opts.options == {'style_opt1': 'A'}
        assert obj1 is obj2

    def test_apply_options_cloned(self):
        obj1 = ExampleElement([])
        obj2 = obj1.options(style_opt1='A')
        opts = Store.lookup_options('backend_1', obj2, 'style')
        assert opts.options == {'style_opt1': 'A'}
        assert obj1 is not obj2

    def test_apply_options_explicit_backend_persist_old_backend(self):
        obj = ExampleElement([])
        obj.opts(style_opt1='A', plot_opt1='B', backend='backend_1')
        obj.opts(style_opt1='C', plot_opt1='D', backend='backend_2')
        plot_opts = Store.lookup_options('backend_1', obj, 'plot')
        assert plot_opts.options == {'plot_opt1': 'B'}
        style_opts = Store.lookup_options('backend_1', obj, 'style')
        assert style_opts.options == {'style_opt1': 'A'}
        plot_opts = Store.lookup_options('backend_2', obj, 'plot')
        assert plot_opts.options == {'plot_opt1': 'D'}
        style_opts = Store.lookup_options('backend_2', obj, 'style')
        assert style_opts.options == {'style_opt1': 'C'}

    def test_apply_options_explicit_backend_persists_other_backend_inverted(self):
        obj = ExampleElement([])
        obj.opts(style_opt1='A', plot_opt1='B', backend='backend_2')
        obj.opts(style_opt1='C', plot_opt1='D', backend='backend_1')
        plot_opts = Store.lookup_options('backend_1', obj, 'plot')
        assert plot_opts.options == {'plot_opt1': 'D'}
        style_opts = Store.lookup_options('backend_1', obj, 'style')
        assert style_opts.options == {'style_opt1': 'C'}
        plot_opts = Store.lookup_options('backend_2', obj, 'plot')
        assert plot_opts.options == {'plot_opt1': 'B'}
        style_opts = Store.lookup_options('backend_2', obj, 'style')
        assert style_opts.options == {'style_opt1': 'A'}

    def test_apply_options_when_backend_switched(self):
        obj = ExampleElement([])
        Store.current_backend = 'backend_2'
        obj.opts(style_opt1='A', plot_opt1='B')
        Store.current_backend = 'backend_1'
        obj.opts(style_opt1='C', plot_opt1='D', backend='backend_2')
        plot_opts = Store.lookup_options('backend_2', obj, 'plot')
        assert plot_opts.options == {'plot_opt1': 'D'}
        style_opts = Store.lookup_options('backend_2', obj, 'style')
        assert style_opts.options == {'style_opt1': 'C'}



class TestOptionsCleanup(CustomBackendTestCase):

    def test_opts_resassignment_cleans_unused_tree(self):
        obj = ExampleElement([]).opts(style_opt1='A').opts(plot_opt1='B')
        custom_options = Store._custom_options['backend_1']
        self.assertIn(obj.id, custom_options)
        self.assertEqual(len(custom_options), 1)

    def test_opts_multiple_resassignment_cleans_unused_tree(self):
        obj = HoloMap({0: ExampleElement([]), 1: ExampleElement([])}).opts(style_opt1='A').opts(plot_opt1='B')
        custom_options = Store._custom_options['backend_1']
        self.assertIn(obj.last.id, custom_options)
        self.assertEqual(len(custom_options), 2)
        for o in obj:
            o.id = None
        self.assertEqual(len(custom_options), 0)

    def test_opts_resassignment_cleans_unused_tree_cross_backend(self):
        obj = ExampleElement([]).opts(style_opt1='A').opts(plot_opt1='B', backend='backend_2')
        custom_options = Store._custom_options['backend_1']
        self.assertIn(obj.id, custom_options)
        self.assertEqual(len(custom_options), 1)
        custom_options = Store._custom_options['backend_2']
        self.assertIn(obj.id, custom_options)
        self.assertEqual(len(custom_options), 1)

    def test_garbage_collect_cleans_unused_tree(self):
        obj = ExampleElement([]).opts(style_opt1='A')
        del obj
        gc.collect()
        custom_options = Store._custom_options['backend_1']
        self.assertEqual(len(custom_options), 0)

    def test_partial_garbage_collect_does_not_clear_tree(self):
        obj = HoloMap({0: ExampleElement([]), 1: ExampleElement([])}).opts(style_opt1='A')
        obj.pop(0)
        gc.collect()
        custom_options = Store._custom_options['backend_1']
        self.assertIn(obj.last.id, custom_options)
        self.assertEqual(len(custom_options), 1)
        obj.pop(1)
        gc.collect()
        self.assertEqual(len(custom_options), 0)

    def test_opts_clear_cleans_unused_tree(self):
        ExampleElement([]).opts(style_opt1='A').opts.clear()
        custom_options = Store._custom_options['backend_1']
        self.assertEqual(len(custom_options), 0)

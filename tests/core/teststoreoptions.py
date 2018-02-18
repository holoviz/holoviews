"""
Unit tests of the StoreOptions class used to control custom options on
Store as used by the %opts magic.
"""
import numpy as np
from holoviews import Overlay, Curve, Image
from holoviews.core.options import Store, StoreOptions
from holoviews.element.comparison import ComparisonTestCase
from holoviews import plotting              # noqa Register backends
from nose.plugins.attrib import attr


class TestStoreOptionsMerge(ComparisonTestCase):

    def setUp(self):
        Store.current_backend = 'matplotlib'
        self.expected = {'Image': {'plot': {'fig_size': 150},
                                   'style': {'cmap': 'Blues'}}}

    def test_full_spec_format(self):
        out = StoreOptions.merge_options(['plot', 'style'],
               options={ 'Image':{'plot':dict(fig_size=150),
                                  'style': dict(cmap='Blues')}})
        self.assertEqual(out, self.expected)

    def test_options_partitioned_format(self):
        out = StoreOptions.merge_options(['plot', 'style'],
            options = dict(plot={'Image':dict(fig_size=150)},
                           style={'Image':dict(cmap='Blues')}))
        self.assertEqual(out, self.expected)

    def test_partitioned_format(self):
        out = StoreOptions.merge_options(['plot', 'style'],
            plot={'Image':dict(fig_size=150)},
            style={'Image':dict(cmap='Blues')})
        self.assertEqual(out, self.expected)


@attr(optional=1) # Requires matplotlib
class TestStoreOptionsCall(ComparisonTestCase):
    """
    The __call__ method makes use of most of the functionality in
    StoreOptions.
    """

    def setUp(self):
        Store.current_backend = 'matplotlib'

    def test_overlay_options_partitioned(self):
        """
        The new style introduced in #73
        """
        data = [zip(range(10),range(10)), zip(range(5),range(5))]
        o = Overlay([Curve(c) for c in data])(
            dict(plot={'Curve.Curve':{'show_grid':False}},
                 style={'Curve.Curve':{'color':'k'}}))

        self.assertEqual(Store.lookup_options('matplotlib',
            o.Curve.I, 'plot').kwargs['show_grid'], False)
        self.assertEqual(Store.lookup_options('matplotlib',
            o.Curve.II, 'plot').kwargs['show_grid'], False)
        self.assertEqual(Store.lookup_options('matplotlib',
            o.Curve.I, 'style').kwargs['color'], 'k')
        self.assertEqual(Store.lookup_options('matplotlib',
            o.Curve.II, 'style').kwargs['color'], 'k')

    def test_overlay_options_complete(self):
        """
        Complete specification style.
        """
        data = [zip(range(10),range(10)), zip(range(5),range(5))]
        o = Overlay([Curve(c) for c in data])(
            {'Curve.Curve':{'plot':{'show_grid':True},
                            'style':{'color':'b'}}})

        self.assertEqual(Store.lookup_options('matplotlib',
            o.Curve.I, 'plot').kwargs['show_grid'], True)
        self.assertEqual(Store.lookup_options('matplotlib',
            o.Curve.II, 'plot').kwargs['show_grid'], True)
        self.assertEqual(Store.lookup_options('matplotlib',
            o.Curve.I, 'style').kwargs['color'], 'b')
        self.assertEqual(Store.lookup_options('matplotlib',
            o.Curve.II, 'style').kwargs['color'], 'b')

    def test_layout_options_short_style(self):
        """
        Short __call__ syntax.
        """
        im = Image(np.random.rand(10,10))
        layout = (im + im).opts({'Layout':dict(plot={'hspace':5})})
        self.assertEqual(Store.lookup_options('matplotlib',
            layout, 'plot').kwargs['hspace'], 5)

    def test_layout_options_long_style(self):
        """
        The old (longer) syntax in __call__
        """
        im = Image(np.random.rand(10,10))
        layout = (im + im).opts({'Layout':dict(plot={'hspace':10})})
        self.assertEqual(Store.lookup_options('matplotlib',
            layout, 'plot').kwargs['hspace'], 10)



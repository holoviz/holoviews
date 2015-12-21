"""
Tests of the parsers implemented in ipython.parsers
"""
from holoviews.element.comparison import ComparisonTestCase
from unittest import SkipTest

try:
    import pyparsing     # noqa (import test)
except ImportError:
    raise SkipTest("Required dependencies not satisfied for testing parsers")

from holoviews.ipython.parser import OptsSpec
from holoviews.core.options import Options, Cycle



class OptsSpecPlotOptionsTests(ComparisonTestCase):
    """
    Test the OptsSpec parser works correctly for plot options.
    """

    def test_plot_opts_simple(self):
        line = "Layout [fig_inches=(3,3) title_format='foo bar']"
        expected= {'Layout':
                   {'plot':
                    Options(title_format='foo bar', fig_inches=(3, 3))}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_plot_opts_with_space(self):
        "Space in the tuple, see issue #77"
        line = "Layout [fig_inches=(3, 3) title_format='foo bar']"
        expected= {'Layout':
                   {'plot':
                    Options(title_format='foo bar', fig_inches=(3, 3))}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_plot_opts_simple_explicit(self):
        line = "Layout plot[fig_inches=(3,3) title_format='foo bar']"
        expected= {'Layout':
                   {'plot':
                    Options(title_format='foo bar', fig_inches=(3, 3))}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_plot_opts_with_space_explicit(self):
        line = "Layout plot[fig_inches=(3, 3) title_format='foo bar']"
        expected= {'Layout':
                   {'plot':
                    Options(title_format='foo bar', fig_inches=(3, 3))}}
        self.assertEqual(OptsSpec.parse(line), expected)



class OptsSpecStyleOptionsTests(ComparisonTestCase):

    def test_style_opts_simple(self):
        line = "Layout (string='foo')"
        expected= {'Layout':{
                    'style': Options(string='foo')}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_style_opts_simple_explicit(self):
        line = "Layout style(string='foo')"
        expected= {'Layout':{
                    'style': Options(string='foo')}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_style_opts_intermediate(self):
        line = "Layout (string='foo' test=3, b=True)"
        expected= {'Layout':{
                    'style': Options(string='foo',
                                     test=3,
                                     b=True)}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_style_opts_intermediate_explicit(self):
        line = "Layout style(string='foo' test=3, b=True )"
        expected= {'Layout':{
                    'style': Options(string='foo',
                                     test=3,
                                     b=True)}}
        self.assertEqual(OptsSpec.parse(line), expected)


    def test_style_opts_advanced(self):
        line = "Layout (string='foo' test=3, b=True color=Cycle(values=[1,2]))"
        expected= {'Layout':{
                    'style': Options(string='foo',
                                     test=3,
                                     b=True,
                                     color=Cycle(values=[1,2]))}}
        self.assertEqual(OptsSpec.parse(line), expected)



class OptsNormPlotOptionsTests(ComparisonTestCase):
    """
    Test the OptsSpec parser works correctly for plot options.
    """

    def test_norm_opts_simple_1(self):
        line = "Layout {+axiswise}"
        expected= {'Layout':
                   {'norm': Options(axiswise=True, framewise=False)}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_norm_opts_simple_explicit_1(self):
        line = "Layout norm{+axiswise}"
        expected= {'Layout':
                   {'norm': Options(axiswise=True, framewise=False)}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_norm_opts_simple_2(self):
        line = "Layout {+axiswise +framewise}"
        expected= {'Layout':
                   {'norm': Options(axiswise=True, framewise=True)}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_norm_opts_simple_explicit_2(self):
        line = "Layout norm{+axiswise +framewise}"
        expected= {'Layout':
                   {'norm': Options(axiswise=True, framewise=True)}}
        self.assertEqual(OptsSpec.parse(line), expected)


class OptsSpecCombinedOptionsTests(ComparisonTestCase):

    def test_combined_1(self):
        line = "Layout plot[fig_inches=(3,3) foo='bar baz'] Layout (string='foo')"
        expected= {'Layout':
                   {'plot':
                    Options(foo='bar baz', fig_inches=(3, 3)),
                    'style': Options(string='foo')}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_combined_two_types_1(self):
        line = "Layout plot[fig_inches=(3,3) foo='bar baz'] Image (string='foo')"
        expected= {'Layout':
                   {'plot':
                    Options(foo='bar baz', fig_inches=(3, 3))},
                   'Image': {
                    'style': Options(string='foo')}}
        self.assertEqual(OptsSpec.parse(line), expected)


    def test_combined_two_types_2(self):
        line = "Layout plot[fig_inches=(3, 3)] Image (string='foo') [foo='bar baz']"
        expected= {'Layout':
                   {'plot':
                    Options(fig_inches=(3, 3))},
                   'Image': {
                       'style': Options(string='foo'),
                       'plot': Options(foo='bar baz')}}
        self.assertEqual(OptsSpec.parse(line), expected)

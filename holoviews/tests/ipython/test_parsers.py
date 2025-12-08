"""
Tests of the parsers implemented in ipython.parsers
"""
import pytest

from holoviews.core.options import Cycle, Options
from holoviews.testing import assert_data_equal, assert_dict_equal

from ..utils import optional_dependencies
from .utils import IPythonCase

pyparsing, _ = optional_dependencies("pyparsing")

if pyparsing:
    from holoviews.util.parser import OptsSpec
else:
    pytest.skip("Required dependencies not satisfied for testing parsers", allow_module_level=True)


class OptsSpecPlotOptionsTests(IPythonCase):
    """
    Test the OptsSpec parser works correctly for plot options.
    """

    def test_plot_opts_simple(self):
        line = "Layout [fig_inches=(3,3) title='foo bar']"
        expected= {'Layout':
                   {'plot':
                    Options(title='foo bar', fig_inches=(3, 3))}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_plot_opts_with_space(self):
        "Space in the tuple, see issue #77"
        line = "Layout [fig_inches=(3, 3) title='foo bar']"
        expected= {'Layout':
                   {'plot':
                    Options(title='foo bar', fig_inches=(3, 3))}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_plot_opts_simple_explicit(self):
        line = "Layout plot[fig_inches=(3,3) title='foo bar']"
        expected= {'Layout':
                   {'plot':
                    Options(title='foo bar', fig_inches=(3, 3))}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_plot_opts_with_space_explicit(self):
        line = "Layout plot[fig_inches=(3, 3) title='foo bar']"
        expected= {'Layout':
                   {'plot':
                    Options(title='foo bar', fig_inches=(3, 3))}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_plot_opts_dict_with_space(self):
        line = "Curve [fontsize={'xlabel': 10, 'title': 20}]"
        expected = {'Curve': {'plot': Options(fontsize={'xlabel': 10, 'title': 20})}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_plot_opts_dict_without_space(self):
        line = "Curve [fontsize=dict(xlabel=10,title=20)]"
        expected = {'Curve': {'plot': Options(fontsize={'xlabel': 10, 'title': 20})}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_plot_opts_nested_brackets(self):
        line = "Curve [title=', '.join(('A', 'B'))]"
        expected = {'Curve': {'plot': Options(title='A, B')}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_plot_opts_multiple_paths(self):
        line = "Image Curve [fig_inches=(3, 3) title='foo bar']"
        expected = {'Image':
                    {'plot':
                     Options(title='foo bar', fig_inches=(3, 3))},
                    'Curve':
                    {'plot':
                     Options(title='foo bar', fig_inches=(3, 3))}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_plot_opts_multiple_paths_2(self):
        line = "Image Curve Layout Overlay[fig_inches=(3, 3) title='foo bar']"
        expected = {'Image':
                    {'plot':
                     Options(title='foo bar', fig_inches=(3, 3))},
                    'Curve':
                    {'plot':
                     Options(title='foo bar', fig_inches=(3, 3))},
                    'Layout':
                    {'plot':
                     Options(title='foo bar', fig_inches=(3, 3))},
                    'Overlay':
                    {'plot':
                     Options(title='foo bar', fig_inches=(3, 3))}}
        assert_dict_equal(OptsSpec.parse(line), expected)


class OptsSpecStyleOptionsTests:

    def test_style_opts_simple(self):
        line = "Layout (string='foo')"
        expected= {'Layout':{
                    'style': Options(string='foo')}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_style_opts_simple_explicit(self):
        line = "Layout style(string='foo')"
        expected= {'Layout':{
                    'style': Options(string='foo')}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_style_opts_intermediate(self):
        line = "Layout (string='foo' test=3, b=True)"
        expected= {'Layout':{
                    'style': Options(string='foo',
                                     test=3,
                                     b=True)}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_style_opts_intermediate_explicit(self):
        line = "Layout style(string='foo' test=3, b=True )"
        expected= {'Layout':{
                    'style': Options(string='foo',
                                     test=3,
                                     b=True)}}
        assert_dict_equal(OptsSpec.parse(line), expected)


    def test_style_opts_advanced(self):
        line = "Layout (string='foo' test=3, b=True color=Cycle(values=[1,2]))"
        expected= {'Layout':{
                    'style': Options(string='foo',
                                     test=3,
                                     b=True,
                                     color=Cycle(values=[1,2]))}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_style_opts_dict_with_space(self):
        line = "Curve (fontsize={'xlabel': 10, 'title': 20})"
        expected = {'Curve': {'style': Options(fontsize={'xlabel': 10, 'title': 20})}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_style_opts_dict_without_space(self):
        line = "Curve (fontsize={'xlabel': 10,'title': 20})"
        expected = {'Curve': {'style': Options(fontsize={'xlabel': 10, 'title': 20})}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_style_opts_cycle_function(self):
        # Explicitly compare because list of arrays do not compare correctly
        import numpy as np
        np.random.seed(42)
        line = "Curve (color=Cycle(values=list(np.random.rand(3,3))))"
        options = OptsSpec.parse(line, {'np': np, 'Cycle': Cycle})
        assert 'Curve' in options
        assert 'style' in options['Curve']
        assert 'color' in options['Curve']['style'].kwargs
        assert isinstance(options['Curve']['style'].kwargs['color'], Cycle)
        values = np.array([[ 0.37454012,  0.95071431,  0.73199394],
                           [ 0.59865848,  0.15601864,  0.15599452],
                           [ 0.05808361,  0.86617615,  0.60111501]])
        assert_data_equal(np.array(options['Curve']['style'].kwargs['color'].values),
                         values)

    def test_style_opts_cycle_list(self):
        line = "Curve (color=Cycle(values=['r', 'g', 'b']))"
        expected = {'Curve': {'style': Options(color=Cycle(values=['r', 'g', 'b']))}}
        assert_dict_equal(OptsSpec.parse(line, {'Cycle': Cycle}), expected)

    def test_style_opts_multiple_paths(self):
        line = "Image Curve (color='beautiful')"
        expected = {'Image':
                    {'style':
                     Options(color='beautiful')},
                    'Curve':
                    {'style':
                     Options(color='beautiful')}}
        assert_dict_equal(OptsSpec.parse(line), expected)



class OptsNormPlotOptionsTests:
    """
    Test the OptsSpec parser works correctly for plot options.
    """

    def test_norm_opts_simple_1(self):
        line = "Layout {+axiswise}"
        expected= {'Layout':
                   {'norm': Options(axiswise=True, framewise=False)}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_norm_opts_simple_explicit_1(self):
        line = "Layout norm{+axiswise}"
        expected= {'Layout':
                   {'norm': Options(axiswise=True, framewise=False)}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_norm_opts_simple_2(self):
        line = "Layout {+axiswise +framewise}"
        expected= {'Layout':
                   {'norm': Options(axiswise=True, framewise=True)}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_norm_opts_simple_explicit_2(self):
        line = "Layout norm{+axiswise +framewise}"
        expected= {'Layout':
                   {'norm': Options(axiswise=True, framewise=True)}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_norm_opts_multiple_paths(self):
        line = "Image Curve {+axiswise +framewise}"
        expected = {'Image':
                    {'norm':
                     Options(axiswise=True, framewise=True)},
                    'Curve':
                    {'norm':
                     Options(axiswise=True, framewise=True)}}
        assert_dict_equal(OptsSpec.parse(line), expected)


class OptsSpecCombinedOptionsTests:

    def test_combined_1(self):
        line = "Layout plot[fig_inches=(3,3) foo='bar baz'] Layout (string='foo')"
        expected= {'Layout':
                   {'plot':
                    Options(foo='bar baz', fig_inches=(3, 3)),
                    'style': Options(string='foo')}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_combined_two_types_1(self):
        line = "Layout plot[fig_inches=(3,3) foo='bar baz'] Image (string='foo')"
        expected= {'Layout':
                   {'plot':
                    Options(foo='bar baz', fig_inches=(3, 3))},
                   'Image': {
                    'style': Options(string='foo')}}
        assert_dict_equal(OptsSpec.parse(line), expected)


    def test_combined_two_types_2(self):
        line = "Layout plot[fig_inches=(3, 3)] Image (string='foo') [foo='bar baz']"
        expected= {'Layout':
                   {'plot': Options(fig_inches=(3, 3,))},
                   'Image': {
                       'style': Options(string='foo'),
                       'plot': Options(foo='bar baz'),
                   }}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_combined_multiple_paths(self):
        line = "Image Curve {+framewise} [fig_inches=(3, 3) title='foo bar'] (c='b') Layout [string='foo'] Overlay"
        expected = {'Image':
                    {'norm':
                     Options(framewise=True, axiswise=False),
                     'plot':
                     Options(title='foo bar', fig_inches=(3, 3)),
                     'style':
                     Options(c='b')},
                    'Curve':
                    {'norm':
                     Options(framewise=True, axiswise=False),
                     'plot':
                     Options(title='foo bar', fig_inches=(3, 3)),
                     'style':
                     Options(c='b')},
                    'Layout':
                    {'plot':
                     Options(string='foo')},
                    'Overlay':
                    {}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_combined_multiple_paths_merge(self):
        line = "Image Curve [fig_inches=(3, 3)] (c='b') Image (s=3)"
        expected = {'Image':
                    {'plot':
                     Options(fig_inches=(3, 3)),
                     'style':
                     Options(c='b', s=3)},
                    'Curve':
                    {'plot':
                     Options(fig_inches=(3, 3)),
                    'style':
                    Options(c='b')}}
        assert_dict_equal(OptsSpec.parse(line), expected)

    def test_combined_multiple_paths_merge_precedence(self):
        line = "Image (s=0, c='b') Image (s=3)"
        expected = {'Image':
                    {'style':
                     Options(c='b', s=3)}}
        assert_dict_equal(OptsSpec.parse(line), expected)

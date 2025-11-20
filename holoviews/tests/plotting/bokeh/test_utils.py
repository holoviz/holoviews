import pytest
from bokeh.models import Tool

import holoviews as hv
from holoviews.core import Store
from holoviews.plotting.bokeh.styles import expand_batched_style
from holoviews.plotting.bokeh.util import (
    TOOL_TYPES,
    filter_batched_data,
    glyph_order,
    select_legends,
)

bokeh_renderer = Store.renderers['bokeh']


class TestBokehUtilsInstantiation:

    def test_expand_style_opts_simple(self):
        style = {'line_width': 3}
        opts = ['line_width']
        data, mapping = expand_batched_style(style, opts, {}, nvals=3)
        assert data['line_width'] == [3, 3, 3]
        assert mapping == {'line_width': {'field': 'line_width'}}

    def test_expand_style_opts_multiple(self):
        style = {'line_color': 'red', 'line_width': 4}
        opts = ['line_color', 'line_width']
        data, mapping = expand_batched_style(style, opts, {}, nvals=3)
        assert data['line_color'] == ['red', 'red', 'red']
        assert data['line_width'] == [4, 4, 4]
        assert mapping == {'line_color': {'field': 'line_color'},
                                   'line_width': {'field': 'line_width'}}

    def test_expand_style_opts_line_color_and_color(self):
        style = {'fill_color': 'red', 'color': 'blue'}
        opts = ['color', 'line_color', 'fill_color']
        data, mapping = expand_batched_style(style, opts, {}, nvals=3)
        assert data['line_color'] == ['blue', 'blue', 'blue']
        assert data['fill_color'] == ['red', 'red', 'red']
        assert mapping == {'line_color': {'field': 'line_color'},
                                   'fill_color': {'field': 'fill_color'}}

    def test_expand_style_opts_line_alpha_and_alpha(self):
        style = {'fill_alpha': 0.5, 'alpha': 0.2}
        opts = ['alpha', 'line_alpha', 'fill_alpha']
        data, mapping = expand_batched_style(style, opts, {}, nvals=3)
        assert data['line_alpha'] == [0.2, 0.2, 0.2]
        assert data['fill_alpha'] == [0.5, 0.5, 0.5]
        assert mapping == {'line_alpha': {'field': 'line_alpha'},
                                   'fill_alpha': {'field': 'fill_alpha'}}

    def test_expand_style_opts_color_predefined(self):
        style = {'fill_color': 'red'}
        opts = ['color', 'line_color', 'fill_color']
        data, mapping = expand_batched_style(style, opts, {'color': 'color'}, nvals=3)
        assert data['fill_color'] == ['red', 'red', 'red']
        assert mapping == {'fill_color': {'field': 'fill_color'}}

    def test_filter_batched_data(self):
        data = {'line_color': ['red', 'red', 'red']}
        mapping = {'line_color': 'line_color'}
        filter_batched_data(data, mapping)
        assert data == {}
        assert mapping == {'line_color': 'red'}

    def test_filter_batched_data_as_field(self):
        data = {'line_color': ['red', 'red', 'red']}
        mapping = {'line_color': {'field': 'line_color'}}
        filter_batched_data(data, mapping)
        assert data == {}
        assert mapping == {'line_color': 'red'}

    def test_filter_batched_data_heterogeneous(self):
        data = {'line_color': ['red', 'red', 'blue']}
        mapping = {'line_color': {'field': 'line_color'}}
        filter_batched_data(data, mapping)
        assert data == {'line_color': ['red', 'red', 'blue']}
        assert mapping == {'line_color': {'field': 'line_color'}}

    def test_glyph_order(self):
        order = glyph_order(['scatter_1', 'patch_1', 'rect_1'],
                            ['scatter', 'patch'])
        assert order == ['scatter_1', 'patch_1', 'rect_1']

@pytest.mark.usefixtures("bokeh_backend")
@pytest.mark.parametrize(
    ('figure_index', 'expected'),
    [
        (0, [True, False]),
        (1, [False, True]),
        ([0], [True, False]),
        ([1], [False, True]),
        ([0, 1], [True, True]),
        (True, [True, True]),
        (False, [False, False]),
        (None, [True, False]),
    ],
    ids=["int0", "int1", "list0", "list1", "list01", "True", "False", "None"],
)
def test_select_legends_figure_index(figure_index, expected):
    overlays = [
        hv.Curve([0, 0]) * hv.Curve([1, 1]),
        hv.Curve([2, 2]) * hv.Curve([3, 3]),
    ]
    layout = hv.Layout(overlays)
    select_legends(layout, figure_index)
    output = [ol.opts["show_legend"] for ol in overlays]
    assert expected == output


def test_bokeh_tools_types():
    bk_tools = Tool._known_aliases
    assert len(bk_tools) == len(TOOL_TYPES)
    assert sorted(bk_tools) == sorted(TOOL_TYPES)

    for key in bk_tools:
        assert isinstance(bk_tools[key](), TOOL_TYPES[key])

import pytest
from bokeh.models import Tool, tools as bk_tools

import holoviews as hv
from holoviews.plotting.bokeh.styles import expand_batched_style
from holoviews.plotting.bokeh.util import (
    TOOL_TYPES,
    filter_batched_data,
    get_tool_id,
    glyph_order,
    select_legends,
)

bokeh_renderer = hv.Store.renderers['bokeh']


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
    bk_tools_aliases = Tool._known_aliases
    assert len(bk_tools_aliases) == len(TOOL_TYPES)
    assert sorted(bk_tools_aliases) == sorted(TOOL_TYPES)

    for key in bk_tools_aliases:
        assert isinstance(bk_tools_aliases[key](), TOOL_TYPES[key])

class TestGetToolId:

    def test_string_wheel_zoom(self):
        tool_type, ident = get_tool_id('wheel_zoom')
        assert tool_type is bk_tools.WheelZoomTool
        assert ident == 'both'

    def test_string_xwheel_zoom(self):
        tool_type, ident = get_tool_id('xwheel_zoom')
        assert tool_type is bk_tools.WheelZoomTool
        assert ident == 'width'

    def test_string_ywheel_zoom(self):
        tool_type, ident = get_tool_id('ywheel_zoom')
        assert tool_type is bk_tools.WheelZoomTool
        assert ident == 'height'

    def test_string_xwheel_pan(self):
        tool_type, ident = get_tool_id('xwheel_pan')
        assert tool_type is bk_tools.WheelPanTool
        assert ident == 'width'

    def test_string_ywheel_pan(self):
        tool_type, ident = get_tool_id('ywheel_pan')
        assert tool_type is bk_tools.WheelPanTool
        assert ident == 'height'

    def test_string_tap(self):
        tool_type, ident = get_tool_id('tap')
        assert tool_type is bk_tools.TapTool
        assert ident == 'tap'

    def test_string_click(self):
        tool_type, ident = get_tool_id('click')
        assert tool_type is bk_tools.TapTool
        assert ident == 'inspect'

    def test_string_doubletap(self):
        tool_type, ident = get_tool_id('doubletap')
        assert tool_type is bk_tools.TapTool
        assert ident == 'doubletap'

    def test_instance_wheel_zoom_default(self):
        tool = bk_tools.WheelZoomTool()
        tool_type, ident = get_tool_id(tool)
        assert tool_type is bk_tools.WheelZoomTool
        assert ident == 'both'

    def test_instance_wheel_zoom_width(self):
        tool = bk_tools.WheelZoomTool(dimensions='width')
        tool_type, ident = get_tool_id(tool)
        assert tool_type is bk_tools.WheelZoomTool
        assert ident == 'width'

    def test_instance_wheel_pan_width(self):
        tool = bk_tools.WheelPanTool(dimension='width')
        tool_type, ident = get_tool_id(tool)
        assert tool_type is bk_tools.WheelPanTool
        assert ident == 'width'

    def test_instance_save_tool_no_tags(self):
        tool = bk_tools.SaveTool(tags=['hv_created'])
        tool_type, ident = get_tool_id(tool)
        assert tool_type is bk_tools.SaveTool
        assert ident is None

    def test_instance_save_tool_user_tag(self):
        tool = bk_tools.SaveTool(tags=['my_tag'])
        tool_type, ident = get_tool_id(tool)
        assert tool_type is bk_tools.SaveTool
        assert ident == ('my_tag',)

    def test_instance_save_tool_hv_and_user_tag(self):
        tool = bk_tools.SaveTool(tags=['hv_created', 'my_tag'])
        tool_type, ident = get_tool_id(tool)
        assert tool_type is bk_tools.SaveTool
        assert ident == ('my_tag',)

    def test_instance_wheel_zoom_hv_created_tag(self):
        tool = bk_tools.WheelZoomTool(tags=['hv_created'])
        tool_type, ident = get_tool_id(tool)
        assert tool_type is bk_tools.WheelZoomTool
        # dimensions takes precedence over tags
        assert ident == 'both'

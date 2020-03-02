import numpy as np

from ...core.options import Store
from ...core.overlay import NdOverlay
from ...selection import OverlaySelectionDisplay, SelectionDisplay


class TabularSelectionDisplay(SelectionDisplay):

    def _build_selection(self, el, exprs, **kwargs):
        opts = {}
        if exprs[1]:
            opts['selected'] = np.where(exprs[1].apply(el.dataset, expanded=True, flat=True))[0]
        return el.opts(clone=True, backend='bokeh', **opts)

    def build_selection(self, selection_streams, hvobj, operations, region_stream=None):
        sel_streams = [selection_streams.exprs_stream]
        hvobj = hvobj.apply(self._build_selection, streams=sel_streams, per_element=True)
        for op in operations:
            hvobj = op(hvobj)
        return hvobj


class BokehOverlaySelectionDisplay(OverlaySelectionDisplay):
    """
    Overlay selection display subclass for use with bokeh backend
    """

    def _build_element_layer(self, element, layer_color, layer_alpha, **opts):
        backend_options = Store.options(backend='bokeh')
        el_name = type(element).name
        style_options = backend_options[(el_name,)]['style']
        allowed = style_options.allowed_keywords

        merged_opts = {opt_name: layer_alpha for opt_name in allowed
                       if 'alpha' in opt_name}
        if el_name in ('HeatMap', 'QuadMesh'):
            merged_opts = {k: v for k, v in merged_opts.items() if 'line_' not in k}
        elif layer_color is None:
            # Keep current color (including color from cycle)
            for color_prop in self.color_props:
                current_color = element.opts.get(group="style")[0].get(color_prop, None)
                if current_color:
                    merged_opts.update({color_prop: current_color})
        else:
            # set color
            merged_opts.update(self._get_color_kwarg(layer_color))

        for opt in ('cmap', 'colorbar'):
            if opt in opts and opt in allowed:
                merged_opts[opt] = opts[opt]

        filtered = {k: v for k, v in merged_opts.items() if k in allowed}
        return element.opts(backend='bokeh', clone=True, tools=['box_select'],
                            **filtered)

    def _style_region_element(self, region_element, unselected_color):
        from ..util import linear_gradient
        backend_options = Store.options(backend="bokeh")
        if isinstance(region_element, NdOverlay):
            element_name = type(region_element.last).name
        else:
            element_name = type(region_element).name
        style_options = backend_options[(element_name,)]['style']
        allowed = style_options.allowed_keywords
        options = {}
        for opt_name in allowed:
            if 'alpha' in opt_name:
                options[opt_name] = 1.0

        if element_name != "Histogram":
            # Darken unselected color
            if unselected_color:
                region_color = linear_gradient(unselected_color, "#000000", 9)[3]
                options["color"] = region_color
            if element_name == 'Rectangles':
                options["line_width"] = 1
                options["fill_alpha"] = 0
                options["selection_fill_alpha"] = 0
                options["nonselection_fill_alpha"] = 0
            elif "Span" in element_name:
                unselected_color = unselected_color or "#e6e9ec"
                region_color = linear_gradient(unselected_color, "#000000", 9)[1]
                options["color"] = region_color
                options["fill_alpha"] = 0.2
                options["selection_fill_alpha"] = 0.2
                options["nonselection_fill_alpha"] = 0.2
        else:
            # Darken unselected color slightly
            unselected_color = unselected_color or "#e6e9ec"
            region_color = linear_gradient(unselected_color, "#000000", 9)[1]
            options["fill_color"] = region_color
            options["color"] = region_color

        return region_element.opts(element_name, backend='bokeh', clone=True, **options)

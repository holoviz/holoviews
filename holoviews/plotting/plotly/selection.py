from __future__ import absolute_import
from ...selection import OverlaySelectionDisplay
from ...core.options import Store


class PlotlyOverlaySelectionDisplay(OverlaySelectionDisplay):
    """
    Overlay selection display subclass for use with plotly backend
    """
    def _build_element_layer(
            self, element, layer_color, selection_expr=True
    ):
        element = self._select(element, selection_expr)

        backend_options = Store.options(backend='plotly')
        style_options = backend_options[(type(element).name,)]['style']

        if 'selectedpoints' in style_options.allowed_keywords:
            shared_opts = dict(selectedpoints=False)
        else:
            shared_opts = dict()

        merged_opts = dict(shared_opts)

        if layer_color is not None:
            # set color
            merged_opts.update(self._get_color_kwarg(layer_color))
        else:
            # Keep current color (including color from cycle)
            for color_prop in self.color_props:
                current_color = element.opts.get(group="style")[0].get(color_prop, None)
                if current_color:
                    merged_opts.update({color_prop: current_color})

        layer_element = element.options(**merged_opts)

        return layer_element

    def _style_region_element(self, region_element, unselected_color):
        from ..util import linear_gradient
        backend_options = Store.options(backend="plotly")
        element_name = type(region_element).name
        style_options = backend_options[(type(region_element).name,)]['style']
        allowed_keywords = style_options.allowed_keywords
        options = {}

        if element_name != "Histogram":
            # Darken unselected color
            region_color = linear_gradient(unselected_color, "#000000", 9)[3]
            if "line_width" in allowed_keywords:
                options["line_width"] = 1
        else:
            # Darken unselected color slightly
            region_color = linear_gradient(unselected_color, "#000000", 9)[1]

        if "color" in allowed_keywords:
            options["color"] = region_color
        elif "line_color" in allowed_keywords:
            options["line_color"] = region_color

        if "selectedpoints" in allowed_keywords:
            options["selectedpoints"] = False

        return region_element.options(**options)

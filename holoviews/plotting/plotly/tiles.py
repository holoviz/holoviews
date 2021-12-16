from holoviews.plotting.plotly import ElementPlot
from holoviews.plotting.plotly.util import STYLE_ALIASES
import numpy as np
from holoviews.element.tiles import _ATTRIBUTIONS


class TilePlot(ElementPlot):
    style_opts = ['min_zoom', 'max_zoom', "alpha", "accesstoken", "mapboxstyle"]

    _supports_geo = True

    @classmethod
    def trace_kwargs(cls, **kwargs):
        return {'type': 'scattermapbox'}

    def get_data(self, element, ranges, style, **kwargs):
        return [{
            "type": "scattermapbox", "lat": [], "lon": [], "subplot": "mapbox",
            "showlegend": False,
        }]

    def graph_options(self, element, ranges, style, **kwargs):
        style = dict(style)
        opts = dict(
            style=style.pop("mapboxstyle", "white-bg"),
            accesstoken=style.pop("accesstoken", None),
        )
        # Extract URL and lower case wildcard characters for mapbox
        url = element.data
        if url:
            layer = {}
            opts["layers"] = [layer]

            # element.data is xyzservices.TileProvider
            if isinstance(element.data, dict):
                layer["source"] = [element.data.build_url(scale_factor="@2x")]
                layer['sourceattribution'] = element.data.html_attribution
                layer['minzoom'] = element.data.get("min_zoom", 0)
                layer['maxzoom'] = element.data.get("max_zoom", 20)
            else:
                for v in ["X", "Y", "Z"]:
                    url = url.replace("{%s}" % v, "{%s}" % v.lower())
                layer["source"] = [url]

                for key, attribution in _ATTRIBUTIONS.items():
                    if all(k in element.data for k in key):
                        layer['sourceattribution'] = attribution

            layer["below"] = 'traces'
            layer["sourcetype"] = "raster"
            # Remaining style options are layer options
            layer.update({STYLE_ALIASES.get(k, k): v for k, v in style.items()})

        return opts

    def get_extents(self, element, ranges, range_type='combined'):
        extents = super(TilePlot, self).get_extents(element, ranges, range_type)
        if (not self.overlaid and all(e is None or not np.isfinite(e) for e in extents)
            and range_type in ('combined', 'data')):
            x0, x1 = (-20037508.342789244, 20037508.342789244)
            y0, y1 = (-20037508.342789255, 20037508.342789244)
            global_extent = (x0, y0, x1, y1)
            return global_extent
        return extents

    def init_graph(self, datum, options, index=0, **kwargs):
        return {'traces': [datum], "mapbox": options}

    def generate_plot(self, key, ranges, element=None, is_geo=False):
        """
        Override to force is_geo to True
        """
        return super(TilePlot, self).generate_plot(key, ranges, element, is_geo=True)

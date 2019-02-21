from __future__ import absolute_import, division, unicode_literals

import param
import numpy as np

from ..core import util
from ..core.dimension import Dimension
from ..core.element import Element2D


class Tiles(Element2D):
    """
    The Tiles Element represents a Web Map Tile Service specified as a
    URL containing {x}, {y}, and {z} templating variables, e.g.:

    https://maps.wikimedia.org/osm-intl/{Z}/{X}/{Y}@2x.png
    """

    kdims = param.List(default=[Dimension('x'), Dimension('y')],
                       bounds=(2, 2), constant=True, doc="""
        The key dimensions of a geometry represent the x- and y-
        coordinates in a 2D space.""")

    group = param.String(default='Tiles')

    def __init__(self, data, kdims=None, vdims=None, **params):
        try:
            from bokeh.models import MercatorTileSource
        except:
            MercatorTileSource = None
        if MercatorTileSource and isinstance(data, MercatorTileSource):
            data = data.url
        elif not isinstance(data, util.basestring):
            raise TypeError('%s data should be a tile service URL not a %s type.'
                            % (type(self).__name__, type(data).__name__) )
        super(Tiles, self).__init__(data, kdims=kdims, vdims=vdims, **params)


    def range(self, dim, data_range=True, dimension_range=True):
        return np.nan, np.nan

    def dimension_values(self, dimension, expanded=True, flat=True):
        return np.array([])


# Mapping between patterns to match specified as tuples and tuples containing attributions
_ATTRIBUTIONS = {
    ('openstreetmap',) : (
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    ),
    ('cartodb',) : (
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, '
        '&copy; <a href="https://cartodb.com/attributions">CartoDB</a>'
    ),
    ('cartocdn',) : (
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, '
        '&copy; <a href="https://cartodb.com/attributions">CartoDB</a>'
    ),
    ('stamen', 'com/t') : ( # to match both 'toner' and 'terrain'
        'Map tiles by <a href="https://stamen.com">Stamen Design</a>, '
        'under <a href="https://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. '
        'Data by <a href="https://openstreetmap.org">OpenStreetMap</a>, '
        'under <a href="https://www.openstreetmap.org/copyright">ODbL</a>.'
    ),
    ('stamen', 'watercolor') : (
        'Map tiles by <a href="https://stamen.com">Stamen Design</a>, '
        'under <a href="https://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. '
        'Data by <a href="https://openstreetmap.org">OpenStreetMap</a>, '
        'under <a href="https://creativecommons.org/licenses/by-sa/3.0">CC BY SA</a>.'
    ),
    ('wikimedia',) : (
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    ),
    ('arcgis','Terrain') : (
        '&copy; <a href="http://downloads.esri.com/ArcGISOnline/docs/tou_summary.pdf">Esri</a>, '
        'USGS, NOAA'
    ),
    ('arcgis','Reference') : (
        '&copy; <a href="http://downloads.esri.com/ArcGISOnline/docs/tou_summary.pdf">Esri</a>, '
        'Garmin, USGS, NPS'
    ),
    ('arcgis','Imagery') : (
        '&copy; <a href="http://downloads.esri.com/ArcGISOnline/docs/tou_summary.pdf">Esri</a>, '
        'Earthstar Geographics'
    ),
    ('arcgis','NatGeo') : (
        '&copy; <a href="http://downloads.esri.com/ArcGISOnline/docs/tou_summary.pdf">Esri</a>, '
        'NatGeo, Garmin, HERE, UNEP-WCMC, USGS, NASA, ESA, METI, NRCAN, GEBCO, NOAA, Increment P'
    ),
    ('arcgis','USA_Topo') : (
        '&copy; <a href="http://downloads.esri.com/ArcGISOnline/docs/tou_summary.pdf">Esri</a>, '
        'NatGeo, i-cubed'
    )
}

# CartoDB basemaps
CartoDark = Tiles('https://cartodb-basemaps-4.global.ssl.fastly.net/dark_all/{Z}/{X}/{Y}.png', name="CartoDark")
CartoEco = Tiles('http://3.api.cartocdn.com/base-eco/{Z}/{X}/{Y}.png', name="CartoEco")
CartoLight = Tiles('https://cartodb-basemaps-4.global.ssl.fastly.net/light_all/{Z}/{X}/{Y}.png', name="CartoLight")
CartoMidnight = Tiles('http://3.api.cartocdn.com/base-midnight/{Z}/{X}/{Y}.png', name="CartoMidnight")

# Stamen basemaps
StamenTerrain = Tiles('http://tile.stamen.com/terrain/{Z}/{X}/{Y}.png', name="StamenTerrain")
StamenTerrainRetina = Tiles('http://tile.stamen.com/terrain/{Z}/{X}/{Y}@2x.png', name="StamenTerrainRetina")
StamenWatercolor = Tiles('http://tile.stamen.com/watercolor/{Z}/{X}/{Y}.jpg', name="StamenWatercolor")
StamenToner = Tiles('http://tile.stamen.com/toner/{Z}/{X}/{Y}.png', name="StamenToner")
StamenTonerBackground = Tiles('http://tile.stamen.com/toner-background/{Z}/{X}/{Y}.png', name="StamenTonerBackground")
StamenLabels = Tiles('http://tile.stamen.com/toner-labels/{Z}/{X}/{Y}.png', name="StamenLabels")

# Esri maps (see https://server.arcgisonline.com/arcgis/rest/services for the full list)
EsriImagery = Tiles('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{Y}/{X}.jpg', name="EsriImagery")
EsriNatGeo = Tiles('https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{Z}/{Y}/{X}', name="EsriNatGeo")
EsriUSATopo = Tiles('https://server.arcgisonline.com/ArcGIS/rest/services/USA_Topo_Maps/MapServer/tile/{Z}/{Y}/{X}', name="EsriUSATopo")
EsriTerrain = Tiles('https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{Z}/{Y}/{X}', name="EsriTerrain")
EsriReference = Tiles('http://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Reference_Overlay/MapServer/tile/{Z}/{Y}/{X}', name="EsriReference")
ESRI = EsriImagery # For backwards compatibility with gv 1.5


# Miscellaneous
OSM = Tiles('http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png', name="OSM")
Wikipedia = Tiles('https://maps.wikimedia.org/osm-intl/{Z}/{X}/{Y}@2x.png', name="Wikipedia")


tile_sources = {k: v for k, v in locals().items() if isinstance(v, Tiles) and k != 'ESRI'}

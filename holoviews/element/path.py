"""
The path module provides a set of elements to draw paths and polygon
geometries in 2D space. In addition to three general elements are
Path, Contours and Polygons, it defines a number of elements to
quickly draw common shapes.
"""

import numpy as np

import param
from ..core import Element2D, Dataset
from ..core.data import MultiInterface
from ..core.dimension import Dimension, asdim
from ..core.util import OrderedDict, disable_constant, isscalar
from .geom import Geometry


class Path(Geometry):
    """
    The Path element represents one or more of path geometries with
    associated values. Each path geometry may be split into
    sub-geometries on NaN-values and may be associated with scalar
    values or array values varying along its length. In analogy to
    GEOS geometry types a Path is a collection of LineString and
    MultiLineString geometries with associated values.

    Like all other elements a Path may be defined through an
    extensible list of interfaces. Natively, HoloViews provides the
    MultiInterface which allows representing paths as lists of regular
    columnar data objects including arrays, dataframes and
    dictionaries of column arrays and scalars.

    The canonical representation is a list of dictionaries storing the
    x- and y-coordinates along with any other values:

        [{'x': 1d-array, 'y': 1d-array, 'value': scalar, 'continuous': 1d-array}, ...]

    Alternatively Path also supports a single columnar data-structure
    to specify an individual path:

        {'x': 1d-array, 'y': 1d-array, 'value': scalar, 'continuous': 1d-array}

    Both scalar values and values continuously varying along the
    geometries coordinates a Path may be used vary visual properties
    of the paths such as the color. Since not all formats allow
    storing scalar values as actual scalars, arrays that are the same
    length as the coordinates but have only one unique value are also
    considered scalar.

    The easiest way of accessing the individual geometries is using
    the `Path.split` method, which returns each path geometry as a
    separate entity, while the other methods assume a flattened
    representation where all paths are separated by NaN values.
    """

    group = param.String(default="Path", constant=True)

    datatype = param.ObjectSelector(default=[
        'multitabular', 'dataframe', 'dictionary', 'dask', 'array'])

    def __init__(self, data, kdims=None, vdims=None, **params):
        if isinstance(data, tuple) and len(data) == 2:
            # Add support for (x, ys) where ys defines multiple paths
            x, y = map(np.asarray, data)
            if y.ndim > 1:
                if len(x) != y.shape[0]:
                    raise ValueError("Path x and y values must be the same length.")
                data = [np.column_stack((x, y[:, i])) for i in range(y.shape[1])]
        elif isinstance(data, list) and all(isinstance(path, Path) for path in data):
            # Allow unpacking of a list of Path elements
            kdims = kdims or self.kdims
            paths = []
            for path in data:
                if path.kdims != kdims:
                    redim = {okd.name: nkd for okd, nkd in zip(path.kdims, kdims)}
                    path = path.redim(**redim)
                if path.interface.multi and isinstance(path.data, list):
                    paths += path.data
                else:
                    paths.append(path.data)
            data = paths

        datatype = params.pop('datatype', self.datatype)

        # Ensure that a list of tuples of scalars and any other non-list
        # type is interpreted as a single path
        if (not isinstance(data, (list, Dataset)) or
            (isinstance(data, list) and not len(data) == 0 and all(
                isinstance(d, tuple) and all(isscalar(v) for v in d)
                for d in data))):
            datatype = [dt for dt in datatype if dt != 'multitabular']
        elif isinstance(data, list) and 'multitabular' not in datatype:
            datatype = datatype + ['multitabular']

        super(Path, self).__init__(data, kdims=kdims, vdims=vdims,
                                   datatype=datatype, **params)


    def __getitem__(self, key):
        if key in self.dimensions(): return self.dimension_values(key)
        if not isinstance(key, tuple) or len(key) == 1:
            key = (key, slice(None))
        elif len(key) == 0: return self.clone()
        if not all(isinstance(k, slice) for k in key):
            raise KeyError("%s only support slice indexing" %
                             self.__class__.__name__)
        xkey, ykey = key
        xstart, xstop = xkey.start, xkey.stop
        ystart, ystop = ykey.start, ykey.stop
        return self.clone(extents=(xstart, ystart, xstop, ystop))


    def select(self, *args, **kwargs):
        """
        Bypasses selection on data and sets extents based on selection.
        """
        return super(Element2D, self).select(*args, **kwargs)


    def split(self, start=None, end=None, datatype=None, **kwargs):
        """
        The split method allows splitting a Path type into a list of
        subpaths of the same type. A start and/or end may be supplied
        to select a subset of paths.
        """
        if not self.interface.multi:
            if datatype == 'array':
                obj = self.array(**kwargs)
            elif datatype == 'dataframe':
                obj = self.dframe(**kwargs)
            elif datatype == 'columns':
                obj = self.columns(**kwargs)
            elif datatype is None:
                obj = self
            else:
                raise ValueError("%s datatype not support" % datatype)
            return [obj]
        return self.interface.split(self, start, end, datatype, **kwargs)

    # Deprecated methods

    @classmethod
    def collapse_data(cls, data_list, function=None, kdims=None, **kwargs):
        param.main.param.warning(
            'Path.collapse_data is deprecated, collapsing may now '
            'be performed through concatenation and aggregation.')
        if function is None:
            return [path for paths in data_list for path in paths]
        else:
            raise Exception("Path types are not uniformly sampled and"
                            "therefore cannot be collapsed with a function.")


    def __setstate__(self, state):
        """
        Ensures old-style unpickled Path types without an interface
        use the MultiInterface.

        Note: Deprecate as part of 2.0
        """
        self.__dict__ = state
        if 'interface' not in state:
            self.interface = MultiInterface
        super(Dataset, self).__setstate__(state)



class Contours(Path):
    """
    The Contours element is a subtype of a Path which is characterized
    by the fact that each path geometry may only be associated with
    scalar values. It supports all the same data formats as a `Path`
    but does not allow continuously varying values along the path
    geometry's coordinates. Conceptually Contours therefore represent
    iso-contours or isoclines, i.e. a function of two variables which
    describes a curve along which the function has a constant value.

    The canonical representation is a list of dictionaries storing the
    x- and y-coordinates along with any other (scalar) values:

        [{'x': 1d-array, 'y': 1d-array, 'value': scalar}, ...]

    Alternatively Contours also supports a single columnar
    data-structure to specify an individual contour:

        {'x': 1d-array, 'y': 1d-array, 'value': scalar, 'continuous': 1d-array}

    Since not all formats allow storing scalar values as actual
    scalars arrays which are the same length as the coordinates but
    have only one unique value are also considered scalar. This is
    strictly enforced, ensuring that each path geometry represents
    a valid iso-contour.

    The easiest way of accessing the individual geometries is using
    the `Contours.split` method, which returns each path geometry as a
    separate entity, while the other methods assume a flattened
    representation where all paths are separated by NaN values.
    """

    level = param.Number(default=None, doc="""
        Optional level associated with the set of Contours.""")

    vdims = param.List(default=[], constant=True, doc="""
        Contours optionally accept a value dimension, corresponding
        to the supplied values.""")

    group = param.String(default='Contours', constant=True)

    _level_vdim = Dimension('Level') # For backward compatibility

    def __init__(self, data, kdims=None, vdims=None, **params):
        data = [] if data is None else data
        if params.get('level') is not None:
            self.param.warning(
                "The level parameter on %s elements is deprecated, "
                "supply the value dimension(s) as columns in the data.",
                type(self).__name__)
            vdims = vdims or [self._level_vdim]
            params['vdims'] = []
        else:
            params['vdims'] = vdims
        super(Contours, self).__init__(data, kdims=kdims, **params)
        if params.get('level') is not None:
            with disable_constant(self):
                self.vdims = [asdim(d) for d in vdims]
        else:
            all_scalar = all(self.interface.isscalar(self, vdim) for vdim in self.vdims)
            if not all_scalar:
                raise ValueError("All value dimensions on a Contours element must be scalar")

    def dimension_values(self, dim, expanded=True, flat=True):
        dimension = self.get_dimension(dim, strict=True)
        if dimension in self.vdims and self.level is not None:
            if expanded:
                return np.full(len(self), self.level)
            return np.array([self.level])
        return super(Contours, self).dimension_values(dim, expanded, flat)



class Polygons(Contours):
    """
    The Polygons element represents one or more polygon geometries
    with associated scalar values. Each polygon geometry may be split
    into sub-geometries on NaN-values and may be associated with
    scalar values. In analogy to GEOS geometry types a Polygons
    element is a collection of Polygon and MultiPolygon
    geometries. Polygon geometries are defined as a set of coordinates
    describing the exterior bounding ring and any number of interior
    holes.

    Like all other elements a Polygons element may be defined through
    an extensible list of interfaces. Natively HoloViews provides the
    MultiInterface which allows representing paths as lists of regular
    columnar data objects including arrays, dataframes and
    dictionaries of column arrays and scalars.

    The canonical representation is a list of dictionaries storing the
    x- and y-coordinates, a list-of-lists of arrays representing the
    holes, along with any other values:

        [{'x': 1d-array, 'y': 1d-array, 'holes': list-of-lists-of-arrays, 'value': scalar}, ...]

    Alternatively Polygons also supports a single columnar
    data-structure to specify an individual polygon:

        {'x': 1d-array, 'y': 1d-array, 'holes': list-of-lists-of-arrays, 'value': scalar}

    The list-of-lists format of the holes corresponds to the potential
    for each coordinate array to be split into a multi-geometry
    through NaN-separators. Each sub-geometry separated by the NaNs
    therefore has an unambiguous mapping to a list of holes. If a
    (multi-)polygon has no holes, the 'holes' key may be ommitted.

    Any value dimensions stored on a Polygons geometry must be scalar,
    just like the Contours element. Since not all formats allow
    storing scalar values as actual scalars arrays which are the same
    length as the coordinates but have only one unique value are also
    considered scalar.

    The easiest way of accessing the individual geometries is using
    the `Polygons.split` method, which returns each path geometry as a
    separate entity, while the other methods assume a flattened
    representation where all paths are separated by NaN values.
    """

    group = param.String(default="Polygons", constant=True)

    vdims = param.List(default=[], doc="""
        Polygons optionally accept a value dimension, corresponding
        to the supplied value.""")

    _level_vdim = Dimension('Value')

    # Defines which key the DictInterface uses to look for holes
    _hole_key = 'holes'

    @property
    def has_holes(self):
        """
        Detects whether any polygon in the Polygons element defines
        holes. Useful to avoid expanding Polygons unless necessary.
        """
        return self.interface.has_holes(self)

    def holes(self):
        """
        Returns a list-of-lists-of-lists of hole arrays. The three levels
        of nesting reflects the structure of the polygons:

          1. The first level of nesting corresponds to the list of geometries
          2. The second level corresponds to each Polygon in a MultiPolygon
          3. The third level of nesting allows for multiple holes per Polygon
        """
        return self.interface.holes(self)


class BaseShape(Path):
    """
    A BaseShape is a Path that can be succinctly expressed by a small
    number of parameters instead of a full path specification. For
    instance, a circle may be expressed by the center position and
    radius instead of an explicit list of path coordinates.
    """

    __abstract = True

    def __init__(self, **params):
        super(BaseShape, self).__init__([], **params)
        self.interface = MultiInterface

    def clone(self, *args, **overrides):
        """
        Returns a clone of the object with matching parameter values
        containing the specified args and kwargs.
        """
        link = overrides.pop('link', True)
        settings = dict(self.get_param_values(), **overrides)
        if 'id' not in settings:
            settings['id'] = self.id
        if not args and link:
            settings['plot_id'] = self._plot_id

        pos_args = getattr(self, '_' + type(self).__name__ + '__pos_params', [])
        return self.__class__(*(settings[n] for n in pos_args),
                              **{k:v for k,v in settings.items()
                                 if k not in pos_args})



class Box(BaseShape):
    """
    Draw a centered box of a given width at the given position with
    the specified aspect ratio (if any).
    """

    x = param.Number(default=0, doc="The x-position of the box center.")

    y = param.Number(default=0, doc="The y-position of the box center.")

    width = param.Number(default=1, doc="The width of the box.")

    height = param.Number(default=1, doc="The height of the box.")

    orientation = param.Number(default=0, doc="""
       Orientation in the Cartesian coordinate system, the
       counterclockwise angle in radians between the first axis and the
       horizontal.""")

    aspect= param.Number(default=1.0, doc="""
       Optional multiplier applied to the box size to compute the
       width in cases where only the length value is set.""")

    group = param.String(default='Box', constant=True, doc="The assigned group name.")

    __pos_params = ['x','y', 'height']

    def __init__(self, x, y, spec, **params):

        if isinstance(spec, tuple):
            if 'aspect' in params:
                raise ValueError('Aspect parameter not supported when supplying '
                                 '(width, height) specification.')
            (width, height ) = spec
        else:
            width, height = params.get('width', spec), spec

        params['width']=params.get('width',width)
        super(Box, self).__init__(x=x, y=y, height=height, **params)

        half_width = (self.width * self.aspect)/ 2.0
        half_height = self.height / 2.0
        (l,b,r,t) = (x-half_width, y-half_height, x+half_width, y+half_height)

        box = np.array([(l, b), (l, t), (r, t), (r, b),(l, b)])
        rot = np.array([[np.cos(self.orientation), -np.sin(self.orientation)],
                        [np.sin(self.orientation), np.cos(self.orientation)]])

        self.data = [np.tensordot(rot, box.T, axes=[1,0]).T]


class Ellipse(BaseShape):
    """
    Draw an axis-aligned ellipse at the specified x,y position with
    the given orientation.

    The simplest (default) Ellipse is a circle, specified using:

    Ellipse(x,y, diameter)

    A circle is a degenerate ellipse where the width and height are
    equal. To specify these explicitly, you can use:

    Ellipse(x,y, (width, height))

    There is also an aspect parameter allowing you to generate an ellipse
    by specifying a multiplicating factor that will be applied to the
    height only.

    Note that as a subclass of Path, internally an Ellipse is a
    sequence of (x,y) sample positions. Ellipse could also be
    implemented as an annotation that uses a dedicated ellipse artist.
    """
    x = param.Number(default=0, doc="The x-position of the ellipse center.")

    y = param.Number(default=0, doc="The y-position of the ellipse center.")

    width = param.Number(default=1, doc="The width of the ellipse.")

    height = param.Number(default=1, doc="The height of the ellipse.")

    orientation = param.Number(default=0, doc="""
       Orientation in the Cartesian coordinate system, the
       counterclockwise angle in radians between the first axis and the
       horizontal.""")

    aspect= param.Number(default=1.0, doc="""
       Optional multiplier applied to the diameter to compute the width
       in cases where only the diameter value is set.""")

    samples = param.Number(default=100, doc="The sample count used to draw the ellipse.")

    group = param.String(default='Ellipse', constant=True, doc="The assigned group name.")

    __pos_params = ['x','y', 'height']

    def __init__(self, x, y, spec, **params):

        if isinstance(spec, tuple):
            if 'aspect' in params:
                raise ValueError('Aspect parameter not supported when supplying '
                                 '(width, height) specification.')
            (width, height) = spec
        else:
            width, height = params.get('width', spec), spec

        params['width']=params.get('width',width)
        super(Ellipse, self).__init__(x=x, y=y, height=height, **params)
        angles = np.linspace(0, 2*np.pi, self.samples)
        half_width = (self.width * self.aspect)/ 2.0
        half_height = self.height / 2.0
        #create points
        ellipse = np.array(
            list(zip(half_width*np.sin(angles),
                     half_height*np.cos(angles))))
        #rotate ellipse and add offset
        rot = np.array([[np.cos(self.orientation), -np.sin(self.orientation)],
               [np.sin(self.orientation), np.cos(self.orientation)]])
        self.data = [np.tensordot(rot, ellipse.T, axes=[1,0]).T+np.array([x,y])]


class Bounds(BaseShape):
    """
    An arbitrary axis-aligned bounding rectangle defined by the (left,
    bottom, right, top) coordinate positions.

    If supplied a single real number as input, this value will be
    treated as the radius of a square, zero-center box which will be
    used to compute the corresponding lbrt tuple.
    """

    lbrt = param.Tuple(default=(-0.5, -0.5, 0.5, 0.5), doc="""
          The (left, bottom, right, top) coordinates of the bounding box.""")

    group = param.String(default='Bounds', constant=True, doc="The assigned group name.")

    __pos_params = ['lbrt']

    def __init__(self, lbrt, **params):
        if not isinstance(lbrt, tuple):
            lbrt = (-lbrt, -lbrt, lbrt, lbrt)

        super(Bounds, self).__init__(lbrt=lbrt, **params)
        (l,b,r,t) = self.lbrt
        xdim, ydim = self.kdims
        self.data = [OrderedDict([(xdim.name, np.array([l, l, r, r, l])),
                                  (ydim.name, np.array([b, t, t, b, b]))])]

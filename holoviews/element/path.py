"""
A Path element is a way of drawing arbitrary shapes that can be
overlayed on top of other elements.

Subclasses of Path are designed to generate certain common shapes
quickly and conveniently. For instance, the Box path is often useful
for marking areas of a raster image.

Contours is also a subclass of Path but in addition to simply
displaying some information, there is a numeric value associated with
each collection of paths.
"""

import numpy as np

import param
from ..core import Dimension, Element2D, Dataset
from ..core.data import MultiInterface, ArrayInterface
from ..core.util import disable_constant


class Path(Dataset, Element2D):
    """
    The Path Element contains a list of Paths stored as tabular data
    types including arrays, dataframes and dictionary of column
    arrays. In addition a number of convenient constructors are
    supported:

    1) A list of lists containing x/y coordinate tuples.
    2) A tuple containing an array of length N with the x-values and a
       second array of shape NxP, where P is the number of paths.
    3) A list of tuples each containing arrays x and y values.

    A Path can be split into subpaths using the split method or combined
    into a flat view using the dimension_values, table, and dframe methods,
    where each path is separated by a NaN value.
    """

    kdims = param.List(default=[Dimension('x'), Dimension('y')],
                       constant=True, bounds=(2, 2), doc="""
        The label of the x- and y-dimension of the Image in form
        of a string or dimension object.""")

    group = param.String(default="Path", constant=True)

    datatype = param.ObjectSelector(default=['multitabular'])

    def __init__(self, data, kdims=None, vdims=None, **params):
        if isinstance(data, tuple) and len(data) == 2:
            x, y = map(np.asarray, data)
            if y.ndim == 1:
                y = np.atleast_2d(y).T
            if len(x) != y.shape[0]:
                raise ValueError("Path x and y values must be the same length.")
            data = [np.column_stack((x, y[:, i])) for i in range(y.shape[1])]
        elif isinstance(data, list) and all(isinstance(path, Path) for path in data):
            data = [p for path in data for p in path.data]
        super(Path, self).__init__(data, kdims=kdims, vdims=vdims, **params)

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


    def select(self, selection_specs=None, **kwargs):
        """
        Bypasses selection on data and sets extents based on selection.
        """
        return super(Element2D, self).select(selection_specs, **kwargs)


    @classmethod
    def collapse_data(cls, data_list, function=None, kdims=None, **kwargs):
        if function is None:
            return [path for paths in data_list for path in paths]
        else:
            raise Exception("Path types are not uniformly sampled and"
                            "therefore cannot be collapsed with a function.")

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


class Contours(Path):
    """
    Contours is a type of Path that is also associated with a value
    (the contour level).
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
            self.warning("The level parameter on %s elements is deprecated, "
                         "supply the value dimension(s) as columns in the data.",
                         type(self).__name__)
            vdims = vdims or [self._level_vdim]
            params['vdims'] = []
        else:
            params['vdims'] = vdims
        super(Contours, self).__init__(data, kdims=kdims, **params)
        if params.get('level') is not None:
            with disable_constant(self):
                self.vdims = [d if isinstance(d, Dimension) else Dimension(d)
                              for d in vdims]
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
    Polygons is a Path Element type that may contain any number of
    closed paths with an associated value.
    """

    group = param.String(default="Polygons", constant=True)

    vdims = param.List(default=[], doc="""
        Polygons optionally accept a value dimension, corresponding
        to the supplied value.""")

    _level_vdim = Dimension('Value')


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
        settings = dict(self.get_param_values(), **overrides)
        if 'id' not in settings:
            settings['id'] = self.id
        if not args:
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

    lbrt = param.NumericTuple(default=(-0.5, -0.5, 0.5, 0.5), doc="""
          The (left, bottom, right, top) coordinates of the bounding box.""")

    group = param.String(default='Bounds', constant=True, doc="The assigned group name.")


    __pos_params = ['lbrt']
    def __init__(self, lbrt, **params):
        if not isinstance(lbrt, tuple):
            lbrt = (-lbrt, -lbrt, lbrt, lbrt)

        super(Bounds, self).__init__(lbrt=lbrt, **params)
        (l,b,r,t) = self.lbrt
        self.data = [np.array([(l, b), (l, t), (r, t), (r, b),(l, b)])]

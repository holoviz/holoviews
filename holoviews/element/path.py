"""
A Path element is a way of drawing arbitrary shapes that can be
overlayed on top of other elements.

Subclasses of Path are designed to generate certain common shapes
quickly and condeniently. For instance, the Box path is often useful
for marking areas of a raster image.

Contours is also a subclass of Path but in addition to simply
displaying some information, there is a numeric value associated with
each collection of paths.
"""

import numpy as np

import param
from ..core import Dimension, Element2D


class Path(Element2D):
    """

    The input data is a list of paths. Each path may be an Nx2 numpy
    arrays or some input that may be converted to such an array, for
    instance, a list of coordinate tuples.

    Each point in the path array corresponds to an X,Y coordinate
    along the specified path.
    """

    key_dimensions = param.List(default=[Dimension('x'), Dimension('y')],
                                  constant=True, bounds=(2, 2), doc="""
        The label of the x- and y-dimension of the Image in form
        of a string or dimension object.""")

    group = param.String(default="Path")

    def __init__(self, data, **params):
        if not isinstance(data, list):
            raise ValueError("Path data must be a list paths (Nx2 coordinates)")
        elif len(data) >= 1:
            data = [np.array(p) if not isinstance(p, np.ndarray) else p for p in data ]
        super(Path, self).__init__(data, **params)


    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) == 1:
            key = (key, slice(None))
        elif len(key) == 0: return self.clone()
        if not all(isinstance(k, slice) for k in key):
            raise IndexError("%s only support slice indexing" %
                             self.__class__.__name__)
        xkey, ykey = key
        xstart, xstop = xkey.start, xkey.stop
        ystart, ystop = ykey.start, ykey.stop
        return self.clone(extents=(xstart, ystart, xstop, ystop))


    def __len__(self):
        return len(self.data)


    def dimension_values(self, dimension):
        dim_idx = self.get_dimension_index(dimension)
        if dim_idx >= len(self.dimensions()):
            return super(Path, self).dimension_values(dimension)
        values = []
        for contour in self.data:
            values.append(contour[:, dim_idx])
        return np.concatenate(values) if values else []



class Contours(Path):
    """
    Contours is a type of Path that is also associated with a value
    (the contour level).
    """

    level = param.Number(default=None, doc="""
        Optional level associated with the set of Contours.""")

    value_dimension = param.List(default=[Dimension('Level')], doc="""
        Contours optionally accept a value dimension, corresponding
        to the supplied values.""", bounds=(1,1))

    group = param.String(default='Contours')

    def __init__(self, data, **params):
        data = [] if data is None else data
        super(Contours, self).__init__(data, **params)

    def dimension_values(self, dim):
        dimension = self.get_dimension(dim)
        if dimension in self.value_dimensions:
            return [self.level]
        return super(Contours, self).dimension_values(dim)



class Polygons(Contours):
    """
    Polygons is a Path Element type that may contain any number of
    closed paths with an associated value.
    """

    group = param.String(default="Polygons")

    value_dimensions = param.List(default=[Dimension('Value')], doc="""
        Polygons optionally accept a value dimension, corresponding
        to the supplied value.""", bounds=(1,1))



class BaseShape(Path):
    """
    A BaseShape is a Path that can be succinctly expressed by a small
    number of parameters instead of a full path specification. For
    instance, a circle may be expressed by the center position and
    radius instead of an explicit list of path coordinates.
    """

    __abstract = True

    def clone(self, *args, **overrides):
        """
        Returns a clone of the object with matching parameter values
        containing the specified args and kwargs.
        """
        settings = dict(self.get_param_values(), **overrides)
        return self.__class__(*args, **settings)



class Box(BaseShape):
    """
    Draw a centered box of a given width at the given position with
    the specified aspect ratio (if any).
    """

    x = param.Number(default=0, doc="The x-position of the box center.")

    y = param.Number(default=0, doc="The y-position of the box center.")

    height = param.Number(default=1, doc="The height of the box.")

    aspect= param.Number(default=1, doc=""""
        The aspect ratio of the box if supplied, otherwise an aspect
        of 1.0 is used.""")

    group = param.String(default='Box', doc="The assigned group name.")

    def __init__(self, x, y, height, **params):
        super(Box, self).__init__([], x=x,y =y, height=height, **params)
        width = height * self.aspect
        (l,b,r,t) = (x-width/2.0, y-height/2, x+width/2.0, y+height/2)
        self.data = [np.array([(l, b), (l, t), (r, t), (r, b),(l, b)])]


class Ellipse(BaseShape):
    """
    Draw an axis-aligned ellipse at the specified x,y position with
    the given width and aspect ratio. By default draws a circle
    (aspect=1).

    Note that as a subclass of Path, internally an Ellipse is a
    sequency of (x,y) sample positions. Ellipse could also be
    implemented as an annotation that uses a more appropriate
    matplotlib artist.
    """
    x = param.Number(default=0, doc="The x-position of the ellipse center.")

    y = param.Number(default=0, doc="The y-position of the ellipse center.")

    height = param.Number(default=1, doc="The height of the ellipse.")

    aspect= param.Number(default=1.0, doc="The aspect ratio of the ellipse.")

    samples = param.Number(default=100, doc="The sample count used to draw the ellipse.")

    group = param.String(default='Ellipse', doc="The assigned group name.")

    def __init__(self, x, y, height, **params):
        super(Ellipse, self).__init__([], x=x, y=y, height=height, **params)
        angles = np.linspace(0, 2*np.pi, self.samples)
        radius = height / 2.0
        self.data = [np.array(
            list(zip(radius*self.aspect*np.sin(angles)+x,
                     radius*np.cos(angles)+y)))]


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

    group = param.String(default='Bounds', doc="The assigned group name.")

    def __init__(self, lbrt, **params):
        if not isinstance(lbrt, tuple):
            lbrt = (-lbrt, -lbrt, lbrt, lbrt)

        super(Bounds, self).__init__([], lbrt=lbrt, **params)
        (l,b,r,t) = self.lbrt
        self.data = [np.array([(l, b), (l, t), (r, t), (r, b),(l, b)])]

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
from ..core import Dimension, Element, Element2D


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
        The label of the x- and y-dimension of the Matrix in form
        of a string or dimension object.""")


    def __init__(self, data, **params):
        if not isinstance(data, list):
            raise ValueError("Path data must be a list paths (Nx2 coordinates)")
        elif len(data) >= 1:
            data = [np.array(p) if not isinstance(p, np.ndarray) else p for p in data ]
        super(Path, self).__init__(data, **params)


    def __len__(self):
        return len(self.data)

    @property
    def xlim(self):
        if self._xlim: return self._xlim
        elif len(self):
            xmin = min(min(c[:, 0]) for c in self.data)
            xmax = max(max(c[:, 0]) for c in self.data)
            return xmin, xmax
        else:
            return None

    @property
    def ylim(self):
        if self._ylim: return self._ylim
        elif len(self):
            ymin = min(min(c[:, 0]) for c in self.data)
            ymax = max(max(c[:, 0]) for c in self.data)
            return ymin, ymax
        else:
            return None

    def dimension_values(self, dimension):
        dim_idx = self.get_dimension_index(dimension)
        if dim_idx >= len(self.dimensions()):
            raise KeyError('Dimension %s not found' % str(dimension))
        values = []
        for contour in self.data:
            values.append(contour[:, dim_idx])
        return np.concatenate(values)



class Contours(Path):
    """
    Contours is a type of Path that is also associated with a value
    (the contour level).
    """

    level = param.Number(default=0.5, doc="""
        Optional level associated with the set of Contours.""")

    value_dimension = param.List(default=[Dimension('Level')], doc="""
        Contours optionally accept a value dimension, corresponding
        to the supplied values.""", bounds=(1,1))

    value = param.String(default='Contours')

    def __init__(self, data, **params):
        data = [] if data is None else data
        super(Contours, self).__init__(data, **params)



class Box(Path):
    """
    Draw a centered square of a given dimension or an arbitrary
    rectangle with the specified (left, bottom, right, top)
    coordinates.
    """

    def __init__(self, data, **params):
        if not isinstance(data, (tuple, float)):
            raise ValueError("Input to Box must be either a tuple of format (l,b,r,t) or a radius")
        elif isinstance(data, float):
            data = (-data, -data, data, data)

        (l,b,r,t) = data
        box = np.array([(l, b), (l, t), (r, t), (r, b),(l, b)])
        super(Box, self).__init__([box], **params)



class Ellipse(Path):
    """
    Draw an axis-aligned ellipse at the specified x,y position with
    the given radius and aspect. By default draws an elipse with an
    aspect of 2.

    Note that as a subclass of Path, internally an Ellipse is a
    sequency of (x,y) sample positions. Ellipse could also be
    implemented as an annotation that uses a more appropriate
    matplotlib artist.
    """

    def __init__(self, x, y, radius, aspect=1, samples=100, **params):

        angles = np.linspace(0, 2*np.pi, samples)
        ellipse = np.array(
            list(zip(radius*np.sin(angles)+x,
                     radius*aspect*np.cos(angles)+y)))
        super(Ellipse, self).__init__([ellipse], **params)

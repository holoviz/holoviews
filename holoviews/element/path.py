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


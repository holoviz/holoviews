import numpy as np

import param

from ..core import Dimension, Element, Element2D


class Annotation(Element2D):
    """
    An Annotation is a special type of element that is designed to be
    overlaid on top of any arbitrary 2D element. Annotations have
    neither key nor value dimensions allowing them to be overlaid over
    any type of data.

    Note that one or more Annotations *can* be displayed without being
    overlaid on top of any other data. In such instances (by default)
    they will be displayed using the unit axis limits (0.0-1.0 in both
    directions) unless an explicit 'extents' parameter is
    supplied. The extents of the bottom Annotation in the Overlay is
    used when multiple Annotations are displayed together.
    """

    value = param.String(default='Annotation')

    xlabel, ylabel = "", ""

    def __init__(self, data, **params):
        super(Annotation, self).__init__(data, **params)



class VLine(Annotation):
    "Vertical line annotation at the given position"

    value = param.String(default='VLine')

    def __init__(self, position, **params):
        super(VLine, self).__init__(position, **params)



class HLine(Annotation):
    "Horizontal line annotation at the given position"

    value = param.String(default='HLine')

    def __init__(self, position, **params):
        super(HLine, self).__init__(position, **params)



class Spline(Annotation):
    """
    Draw a spline using the given handle coordinates and handle
    codes. Follows matplotlib spline definitions as used in
    matplotlib.path.Path with the following codes:

    Path.STOP     : 0
    Path.MOVETO   : 1
    Path.LINETO   : 2
    Path.CURVE3   : 3
    Path.CURVE4   : 4
    Path.CLOSEPLOY: 79
    """

    value = param.String(default='Spline')

    def __init__(self, (coords, codes), **params):
        super(Spline, self).__init__((coords, codes), **params)



class Arrow(Annotation):
    """
    Draw an arrow to the given xy position with optional text at
    distance 'points' away. The direction of the arrow may be
    specified as well as the arrow head style.
    """

    value = param.String(default='Arrow')

    def __init__(self, xy, text='', direction='<',
                 points=40, arrowstyle='->', **params):

        directions = ['<', '^', '>', 'v']
        arrowstyles = ['-', '->', '-[', '-|>', '<->', '<|-|>']

        if direction.lower() not in directions:
            raise ValueError("Valid arrow directions are: %s"
                             % ', '.join(repr(d) for d in directions))

        if arrowstyle not in arrowstyles:
            raise ValueError("Valid arrow styles are: %s"
                             % ', '.join(repr(a) for a in arrowstyles))

        info = (direction.lower(), text, xy, points, arrowstyle)
        super(Arrow, self).__init__(info, **params)



class Text(Annotation):
    """
    Draw a text annotation at the specified position with custom
    fontsize, alignment and rotation.
    """

    value = param.String(default='Text')

    def __init__(self, x,y, text, fontsize=12,
                 horizontalalignment='center',
                 verticalalignment='center',
                 rotation=0, **params):
        info = (x,y, text, fontsize,
                horizontalalignment, verticalalignment, rotation)
        super(Text, self).__init__(info, **params)




class Contours(Element2D):
    """
    Allows sets of contour lines to be defined over a
    SheetCoordinateSystem.

    The input data is a list of Nx2 numpy arrays where each array
    corresponds to a contour in the group. Each point in the numpy
    array corresponds to an X,Y coordinate.
    """

    key_dimensions = param.List(default=[Dimension('x'), Dimension('y')],
                                  constant=True, bounds=(2, 2), doc="""
        The label of the x- and y-dimension of the Matrix in form
        of a string or dimension object.""")

    level = param.Number(default=None, doc="""
        Optional level associated with the set of Contours.""")

    value_dimension = param.List(default=[], doc="""
        Contours optionally accept a value dimension, corresponding
        to the supplied values.""", bounds=(0,1))

    value = param.String(default='Contours')

    def __init__(self, data, **params):
        data = [] if data is None else data
        super(Contours, self).__init__(data, **params)
        if self.level and not len(self.value_dimensions):
            self.value_dimensions = [Dimension('Level')]

    def resize(self, bounds):
        return Contours(self.contours, bounds, style=self.style)


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
        if dim_idx >= len(self.dimensions):
            raise KeyError('Dimension %s not found' % str(dimension))
        values = []
        for contour in self.data:
            values.append(contour[:, dim_idx])
        return np.concatenate(values)


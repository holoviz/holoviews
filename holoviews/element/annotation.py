import numpy as np

import param

from ..core import Dimension, Element2D


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

    key_dimensions = param.List(default=[Dimension('x'), Dimension('y')],
                                bounds=(2,2))

    group = param.String(default='Annotation')

    xlabel, ylabel = "", ""

    def __init__(self, data, **params):
        super(Annotation, self).__init__(data, **params)
        self._xlim = (0, 1) if self._xlim is None else self._xlim
        self._ylim = (0, 1) if self._ylim is None else self._ylim

    def dimension_values(self, dimension):
        index = self.get_dimension_index(dimension)
        if index == 0:
            return [self.data if np.isscalar(self.data) else self.data[index]]
        elif index == 1:
            return [] if np.isscalar(self.data) else [self.data[1]]
        else:
            return super(Annotation, self).dimension_values(dimension)


class VLine(Annotation):
    "Vertical line annotation at the given position"

    group = param.String(default='VLine')

    def __init__(self, x, **params):
        super(VLine, self).__init__(x, **params)



class HLine(Annotation):
    "Horizontal line annotation at the given position"

    group = param.String(default='HLine')

    def __init__(self, y, **params):
        super(HLine, self).__init__(y, **params)



class Spline(Annotation):
    """
    Draw a spline using the given handle coordinates and handle
    codes. The constructor accepts a tuple in format (coords, codes).

    Follows matplotlib spline definitions as used in
    matplotlib.path.Path with the following codes:

    Path.STOP     : 0
    Path.MOVETO   : 1
    Path.LINETO   : 2
    Path.CURVE3   : 3
    Path.CURVE4   : 4
    Path.CLOSEPLOY: 79
    """

    group = param.String(default='Spline')

    def __init__(self, spline_points, **params):
        super(Spline, self).__init__(spline_points, **params)



class Arrow(Annotation):
    """
    Draw an arrow to the given xy position with optional text at
    distance 'points' away. The direction of the arrow may be
    specified as well as the arrow head style.
    """

    group = param.String(default='Arrow')

    def __init__(self, x, y, text='', direction='<',
                 points=40, arrowstyle='->', **params):

        directions = ['<', '^', '>', 'v']
        arrowstyles = ['-', '->', '-[', '-|>', '<->', '<|-|>']

        if direction.lower() not in directions:
            raise ValueError("Valid arrow directions are: %s"
                             % ', '.join(repr(d) for d in directions))

        if arrowstyle not in arrowstyles:
            raise ValueError("Valid arrow styles are: %s"
                             % ', '.join(repr(a) for a in arrowstyles))

        info = (direction.lower(), text, (x,y), points, arrowstyle)
        super(Arrow, self).__init__(info, **params)



class Text(Annotation):
    """
    Draw a text annotation at the specified position with custom
    fontsize, alignment and rotation.
    """

    group = param.String(default='Text')

    def __init__(self, x,y, text, fontsize=12,
                 horizontalalignment='center',
                 verticalalignment='center',
                 rotation=0, **params):
        info = (x,y, text, fontsize,
                horizontalalignment, verticalalignment, rotation)
        super(Text, self).__init__(info, **params)

from numbers import Number
import numpy as np
import param

from ..core.util import datetime_types, basestring
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

    kdims = param.List(default=[Dimension('x'), Dimension('y')],
                       bounds=(2,2))

    group = param.String(default='Annotation', constant=True)

    _auxiliary_component = True

    def __init__(self, data, **params):
        super(Annotation, self).__init__(data, **params)

    def __len__(self):
        return 1

    def __getitem__(self, key):
        if key in self.dimensions(): return self.dimension_values(key)
        if not isinstance(key, tuple) or len(key) == 1:
            key = (key, slice(None))
        elif len(key) == 0: return self.clone()
        if not all(isinstance(k, slice) for k in key):
            raise KeyError("%s only support slice indexing" %
                             self.__class__.__name__)
        xkey, ykey = tuple(key[:len(self.kdims)])
        xstart, xstop = xkey.start, xkey.stop
        ystart, ystop = ykey.start, ykey.stop
        return self.clone(self.data, extents=(xstart, ystart, xstop, ystop))


    def dimension_values(self, dimension, expanded=True, flat=True):
        index = self.get_dimension_index(dimension)
        if index == 0:
            return np.array([self.data if np.isscalar(self.data) else self.data[index]])
        elif index == 1:
            return [] if np.isscalar(self.data) else np.array([self.data[1]])
        else:
            return super(Annotation, self).dimension_values(dimension)

    # Note: This version of clone is identical in path.BaseShape
    # Consider implementing a mix-in class if it is needed again.
    def clone(self, *args, **overrides):
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        # Apply name mangling for __ attribute
        pos_args = getattr(self, '_' + type(self).__name__ + '__pos_params', [])
        settings = {k: v for k, v in dict(self.get_param_values(), **overrides).items()
                    if k not in pos_args[:len(args)]}
        if 'id' not in settings:
            settings['id'] = self.id
        return self.__class__(*args, **settings)


class VLine(Annotation):
    "Vertical line annotation at the given position"

    group = param.String(default='VLine', constant=True)

    x = param.ClassSelector(default=0, class_= (Number, ) + datetime_types, doc="""
       The x-position of the VLine which make be numeric or a timestamp.""")

    __pos_params = ['x']

    def __init__(self, x, **params):
        super(VLine, self).__init__(x, x=x, **params)

    def dimension_values(self, dimension, expanded=True, flat=True):
        index = self.get_dimension_index(dimension)
        if index == 0:
            return np.array([self.data])
        elif index == 1:
            return np.array([])
        else:
            return super(VLine, self).dimension_values(dimension)


class HLine(Annotation):
    "Horizontal line annotation at the given position"

    group = param.String(default='HLine', constant=True)

    y = param.ClassSelector(default=0, class_= (Number, ) + datetime_types, doc="""
       The y-position of the VLine which make be numeric or a timestamp.""")

    __pos_params = ['y']

    def __init__(self, y, **params):
        super(HLine, self).__init__(y, y=y, **params)

    def dimension_values(self, dimension, expanded=True, flat=True):
        index = self.get_dimension_index(dimension)
        if index == 0:
            return np.array([])
        elif index == 1:
            return np.array([self.data])
        else:
            return super(HLine, self).dimension_values(dimension)



class Spline(Annotation):
    """
    Draw a spline using the given handle coordinates and handle
    codes. The constructor accepts a tuple in format (coords, codes).

    Follows format of matplotlib spline definitions as used in
    matplotlib.path.Path with the following codes:

    Path.STOP     : 0
    Path.MOVETO   : 1
    Path.LINETO   : 2
    Path.CURVE3   : 3
    Path.CURVE4   : 4
    Path.CLOSEPLOY: 79
    """

    group = param.String(default='Spline', constant=True)

    def __init__(self, spline_points, **params):
        super(Spline, self).__init__(spline_points, **params)


    def dimension_values(self, dimension, expanded=True, flat=True):
        index = self.get_dimension_index(dimension)
        if index in [0, 1]:
            return np.array([point[index] for point in self.data[0]])
        else:
            return super(Spline, self).dimension_values(dimension)



class Arrow(Annotation):
    """
    Draw an arrow to the given xy position with optional text at
    distance 'points' away. The direction of the arrow may be
    specified as well as the arrow head style.
    """

    x = param.ClassSelector(default=0, class_= (Number, ) + datetime_types, doc="""
       The x-position of the arrow which make be numeric or a timestamp.""")

    y = param.ClassSelector(default=0, class_= (Number, ) + datetime_types, doc="""
       The y-position of the arrow which make be numeric or a timestamp.""")

    text = param.String(default='', doc="Text associated with the arrow.")

    direction = param.ObjectSelector(default='<',
                                     objects=['<', '^', '>', 'v'], doc="""
        The cardinal direction in which the arrow is pointing. Accepted
        arrow directions are '<', '^', '>' and 'v'.""")

    arrowstyle = param.ObjectSelector(default='->',
                                      objects=['-', '->', '-[', '-|>', '<->', '<|-|>'],
                                      doc="""
        The arrowstyle used to draw the arrow. Accepted arrow styles are
        '-', '->', '-[', '-|>', '<->' and '<|-|>'""")

    points = param.Number(default=40, doc="Font size of arrow text (if any).")

    group = param.String(default='Arrow', constant=True)

    __pos_params = ['x', 'y', 'text', 'direction', 'points', 'arrowstyle']

    def __init__(self, x, y, text='', direction='<',
                 points=40, arrowstyle='->', **params):

        info = (x, y, text, direction, points, arrowstyle)
        super(Arrow, self).__init__(info, x=x, y=y,
                                    text=text, direction=direction,
                                    points=points, arrowstyle=arrowstyle,
                                    **params)

    def __setstate__(self, d):
        """
        Add compatibility for unpickling old Arrow types with different
        .data format.
        """
        super(Arrow, self).__setstate__(d)
        if len(self.data) == 5:
            direction, text, (x, y), points, arrowstyle = self.data
            self.data = (x, y, text, direction, points, arrowstyle)


    def dimension_values(self, dimension, expanded=True, flat=True):
        index = self.get_dimension_index(dimension)
        if index == 0:
            return np.array([self.x])
        elif index == 1:
            return np.array([self.y])
        else:
            return super(Arrow, self).dimension_values(dimension)



class Text(Annotation):
    """
    Draw a text annotation at the specified position with custom
    fontsize, alignment and rotation.
    """
    x = param.ClassSelector(default=0, class_= (Number, basestring) + datetime_types, doc="""
       The x-position of the arrow which make be numeric or a timestamp.""")

    y = param.ClassSelector(default=0, class_= (Number, basestring) + datetime_types, doc="""
       The y-position of the arrow which make be numeric or a timestamp.""")

    text = param.String(default='', doc="The text to be displayed.")

    fontsize = param.Number(default=12, doc="Font size of the text.")

    rotation = param.Number(default=0, doc="Text rotation angle in degrees.")

    halign= param.ObjectSelector(default='center',
                                 objects= ['left', 'right', 'center'], doc="""
       The horizontal alignment position of the displayed text. Allowed values
       are 'left', 'right' and 'center'.""")

    valign= param.ObjectSelector(default='center',
                                 objects= ['top', 'bottom', 'center'], doc="""
       The vertical alignment position of the displayed text. Allowed values
       are 'center', 'top' and 'bottom'.""")

    group = param.String(default='Text', constant=True)

    __pos_params = ['x', 'y', 'text', 'fontsize', 'halign', 'valign', 'rotation']

    def __init__(self, x,y, text, fontsize=12,
                 halign='center', valign='center', rotation=0, **params):
        info = (x,y, text, fontsize, halign, valign, rotation)
        super(Text, self).__init__(info, x=x, y=y, text=text,
                                   fontsize=fontsize, rotation=rotation,
                                   halign=halign, valign=valign, **params)

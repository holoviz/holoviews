import numpy as np

import param

from ..core import Dimension, Element, Element2D

class Annotation(Element):
    """
    An annotation is a type of ViewableElement that is displayed on the top of an
    overlay. Annotations elements do not depend on the details of the
    data displayed and are generally for the convenience of the user
    (e.g. to draw attention to specific areas of the figure using
    arrows, boxes or labels).

    All annotations have an optional interval argument that indicates
    which map elements they apply to. For instance, this allows
    annotations for a specific time interval when overlaid over a
    HoloMap or HoloMap with a 'Time' dimension. The interval
    argument is a dictionary of dimension keys and tuples containing
    (start, end) values. A value of None, indicates an unspecified
    constraint.
    """

    value = param.String(default='Annotation')

    xlim, ylim = None, None
    xlabel, ylabel = "", ""

    def __init__(self, boxes=[], vlines=[], hlines=[], arrows=[], **params):
        """
        Annotations may be added via method calls or supplied directly
        to the constructor using lists of specification elements or
        (specification, interval) tuples. The specification element
        formats are listed below:

        box: A BoundingBox or ((left, bottom), (right, top)) tuple.

        hline/vline specification: The vertical/horizontal coordinate.

        arrow: An (xy, kwargs) tuple where xy is a coordinate tuple
        and kwargs is a dictionary of the optional arguments accepted
        by the arrow method.
        """
        super(Annotation, self).__init__([], **params)

        for box in boxes:
            if hasattr(box, 'lbrt'):         self.box(box, None)
            elif isinstance(box[1], dict):   self.box(*box)
            else:                            self.box(box, None)

        for vline in vlines:
            self.vline(*(vline if isinstance(vline, tuple) else (vline, None)))

        for hline in hlines:
            self.hline(*(hline if isinstance(hline, tuple) else (hline, None)))

        for arrow in arrows:
            spec, interval = (arrow, None) if isinstance(arrow[0], tuple) else arrow
            self.arrow(spec[0], **dict(spec[1], interval=interval))


    def arrow(self, xy, text='', direction='<', points=40,
              arrowstyle='->', interval=None):
        """
        Draw an arrow along one of the cardinal directions with option
        text. The direction indicates the direction the arrow is
        pointing and the points argument defines the length of the
        arrow in points. Different arrow head styles are supported via
        the arrowstyle argument.
        """
        directions = ['<', '^', '>', 'v']
        if direction.lower() not in directions:
            raise Exception("Valid arrow directions are: %s"
                            % ', '.join(repr(d) for d in directions))

        arrowstyles = ['-', '->', '-[', '-|>', '<->', '<|-|>']
        if arrowstyle not in arrowstyles:
            raise Exception("Valid arrow styles are: %s"
                            % ', '.join(repr(a) for a in arrowstyles))

        self.data.append((direction.lower(), text, xy, points, arrowstyle, interval))


    def line(self, coords, interval=None):
        """
        Draw an arbitrary polyline that goes through the listed
        coordinates.  Coordinates are specified using a list of (x,y)
        tuples.
        """
        self.data.append(('line', coords, interval))


    def spline(self, coords, codes, interval=None):
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
        self.data.append(('spline', coords, codes, interval))


    def box(self, box, interval=None):
        """
        Draw a box with corners specified in the positions specified
        by ((left, bottom), (right, top)). Alternatively, a
        BoundingBox may be supplied.
        """
        if hasattr(box, 'lbrt'):
            (l,b,r,t) = box.lbrt()
        else:
            ((l,b), (r,t)) = box

        self.line(((t,l), (t,r), (b,r), (b,l), (t,l)),
                  interval=interval)


    def vline(self, x, interval=None):
        """
        Draw an axis vline (vertical line) at the given x value.
        """
        self.data.append(('vline', x, interval))


    def hline(self, y, interval=None):
        """
        Draw an axis hline (horizontal line) at the given y value.
        """
        self.data.append(('hline', y, interval))

    def dimension_values(self, dim):
        raise NotImplementedError("Annotation do not have explicit "
                                  "dimension values")

    def hist(self, *args, **kwargs):
        raise NotImplementedError("Annotations don't support histograms.")


    def __mul__(self, other):
        raise Exception("An annotation can only be overlaid over a different ViewableElement type.")


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
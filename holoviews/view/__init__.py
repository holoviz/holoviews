from ..core.view import View

from .dataviews import * # pyflakes:ignore (API import)
from .sheetviews import * # pyflakes:ignore (API import)
from .tabular import * # pyflakes:ignore (API import)


class Annotation(Layer):
    """
    An annotation is a type of View that is displayed on the top of an
    overlay. Annotations elements do not depend on the details of the
    data displayed and are generally for the convenience of the user
    (e.g. to draw attention to specific areas of the figure using
    arrows, boxes or labels).

    All annotations have an optional interval argument that indicates
    which map elements they apply to. For instance, this allows
    annotations for a specific time interval when overlaid over a
    ViewMap or ViewMap with a 'Time' dimension. The interval
    argument is a dictionary of dimension keys and tuples containing
    (start, end) values. A value of None, indicates an unspecified
    constraint.
    """

    xlim, ylim = None, None
    xlabel, ylabel = "", ""

    def __init__(self, boxes=[], vlines=[], hlines=[], arrows=[], **kwargs):
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
        super(Annotation, self).__init__([], **kwargs)

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

    def dim_values(self, dim):
        raise NotImplementedError("Annotation do not have explicit "
                                  "dimension values")

    def hist(self, *args, **kwargs):
        raise NotImplementedError("Annotations don't support histograms.")


    def __mul__(self, other):
        raise Exception("An annotation can only be overlaid over a different View type.")


def public(obj):
    if not isinstance(obj, type): return False
    return issubclass(obj, View)

__all__ = list(set([_k for _k, _v in locals().items() if public(_v)]))

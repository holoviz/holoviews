import param

from .dimension import Dimension
from .holoview import View, HoloMap, find_minmax
from .layout import Element, GridLayout
from .options import channels


class Layer(Element):
    """
    Layer is the baseclass for all 2D View types, with an x- and
    y-dimension. Subclasses should define the data storage in the
    constructor, as well as methods and properties, which define how
    the data maps onto the x- and y- and value dimensions.
    """

    dimensions = param.List(default=[Dimension('X')], doc="""
        Dimensions on Layers determine the number of indexable
        dimensions.""")

    value = param.ClassSelector(class_=Dimension, default=Dimension('Y'))


    def __mul__(self, other):
        if isinstance(other, HoloMap):
            items = [(k, self * v) for (k, v) in other.items()]
            return other.clone(items=items)

        self_layers = self.data if isinstance(self, Overlay) else [self]
        other_layers = other.data if isinstance(other, Overlay) else [other]
        combined_layers = self_layers + other_layers

        return Overlay(combined_layers)


    ########################
    # Subclassable methods #
    ########################


    def __init__(self, data, **kwargs):
        self._xlim = None
        self._ylim = None
        super(Layer, self).__init__(data, **kwargs)


    @property
    def cyclic_range(self):
        if self.dimensions[0].cyclic:
            return self.dimensions[0].range[1]
        else:
            return None

    @property
    def range(self):
        if self.cyclic_range:
            return self.cyclic_range
        y_vals = self.data[:, 1]
        return (float(min(y_vals)), float(max(y_vals)))


    @property
    def xlabel(self):
        return self.dimensions[0].pprint_label


    @property
    def ylabel(self):
        if len(self.dimensions) == 1:
            return self.value.pprint_label
        else:
            return self.dimensions[1].pprint_label

    @property
    def xlim(self):
        if self._xlim:
            return self._xlim
        elif self.cyclic_range is not None:
            return (0, self.cyclic_range)
        else:
            x_vals = self.data[:, 0]
            return (float(min(x_vals)), float(max(x_vals)))

    @xlim.setter
    def xlim(self, limits):
        if self.cyclic_range:
            self.warning('Cannot override the limits of a '
                         'cyclic dimension.')
        elif limits is None or (isinstance(limits, tuple) and len(limits) == 2):
            self._xlim = limits
        else:
            raise ValueError('xlim needs to be a length two tuple or None.')


    @property
    def ylim(self):
        if self._ylim:
            return self._ylim
        else:
            y_vals = self.data[:, 1]
            return (float(min(y_vals)), float(max(y_vals)))


    @ylim.setter
    def ylim(self, limits):
        if limits is None or (isinstance(limits, tuple) and len(limits) == 2):
            self._ylim = limits
        else:
            raise ValueError('xlim needs to be a length two tuple or None.')


    @property
    def lbrt(self):
        l, r = self.xlim if self.xlim else (None, None)
        b, t = self.ylim if self.ylim else (None, None)
        return l, b, r, t


    @lbrt.setter
    def lbrt(self, lbrt):
        l, b, r, t = lbrt
        self.xlim, self.ylim = (l, r), (b, t)



class Overlay(View):
    """
    An Overlay allows a group of Layers to be overlaid together. Layers can
    be indexed out of an overlay and an overlay is an iterable that iterates
    over the contained layers.

    A SheetOverlay may be used to overlay lines or points over a
    SheetMatrix. In addition, if an overlay consists of three or four
    SheetViews of depth 1, the overlay may be converted to an RGB(A)
    SheetMatrix via the rgb property.
    """

    dimensions = param.List(default=[Dimension('Overlay')], constant=True, doc="""List
      of dimensions the View can be indexed by.""")

    label = param.String(doc="""
      A short label used to indicate what kind of data is contained
      within the view object.

      Overlays should not have their label set directly by the user as
      the label is only for defining custom channel operations.""")

    channels = channels

    _abstract = True

    _deep_indexable = True

    def __init__(self, overlays, **kwargs):
        super(Overlay, self).__init__([], **kwargs)
        self._xlim = None
        self._ylim = None
        self._layer_dimensions = None
        self.set(overlays)


    @property
    def labels(self):
        return [el.label for el in self]


    @property
    def style(self):
        return [el.style for el in self.data]


    @style.setter
    def style(self, styles):
        for layer, style in zip(self, styles):
            layer.style = style


    def add(self, layer):
        """
        Overlay a single layer on top of the existing overlay.
        """
        if isinstance(layer, Annotation): pass
        elif not len(self):
            self._layer_dimensions = layer.dimension_labels
            self.xlim = layer.xlim
            self.ylim = layer.ylim
            self.value = layer.value
            self.label = layer.label
        else:
            self.xlim = layer.xlim if self.xlim is None else find_minmax(self.xlim, layer.xlim)
            self.ylim = layer.ylim if self.xlim is None else find_minmax(self.ylim, layer.ylim)
            if layer.dimension_labels != self._layer_dimensions:
                raise Exception("DataLayers must share common dimensions.")
        if layer.label in [o.label for o in self.data]:
            self.warning('Label %s already defined in Overlay' % layer.label)
        self.data.append(layer)


    @property
    def range(self):
        range = self[0].range
        cyclic = self[0].cyclic_range is not None
        for view in self:
            if cyclic != (self[0].cyclic_range is not None):
                raise Exception("Overlay contains cyclic and non-cyclic "
                                "Views, cannot compute range.")
            range = find_minmax(range, view.range)
        return range


    @property
    def cyclic_range(self):
        return self[0].cyclic_range if len(self) else None


    def __add__(self, obj):
        if not isinstance(obj, GridLayout):
            return GridLayout(initial_items=[self, obj])


    def __mul__(self, other):
        if isinstance(other, HoloMap):
            items = [(k, self * v) for (k, v) in other.items()]
            return other.clone(items=items)
        elif isinstance(other, Overlay):
            overlays = self.data + other.data
        elif isinstance(other, (View, Annotation)):
            overlays = self.data + [other]
        else:
            raise TypeError('Can only create an overlay of holoviews.')

        return Overlay(overlays)


    def set(self, layers):
        """
        Set a collection of layers to be overlaid with each other.
        """
        self.data = []
        for layer in layers:
            self.add(layer)
        return self


    def hist(self, index=None, adjoin=True, **kwargs):
        valid_ind = isinstance(index, int) and (0 <= index < len(self))
        valid_label = index in [el.label for el in self.data]
        if index is None or not any([valid_ind, valid_label]):
            raise TypeError("Please supply a suitable index for the histogram data")

        hist = self[index].hist(adjoin=False, **kwargs)
        if adjoin:
            layout = self << hist
            layout.main_layer = index
            return layout
        else:
            return hist


    def __getitem__(self, ind):
        if isinstance(ind, str):
            matches = [o for o in self.data if o.label == ind]
            if matches == []: raise KeyError('Key %s not found.' % ind)
            return matches[0]

        if ind is ():
            return self
        elif isinstance(ind, tuple):
            ind, ind2 = (ind[0], ind[1:])
        else:
            return self.data[ind]
        if isinstance(ind, slice):
            return self.__class__([d[ind2] for d in self.data[ind]],
                                  **dict(self.get_param_values()))
        else:
            return self.data[ind][ind2]


    def __getstate__(self):
        """
        When pickling, make sure to save the relevant channel
        definitions.
        """
        obj_dict = self.__dict__.copy()
        channels = dict((k, self.channels[k]) for k in self.channels.keys())
        obj_dict['channel_definitions'] = channels
        return obj_dict


    def __setstate__(self, d):
        """
        When unpickled, restore the saved channel definitions.
        """

        if 'channel_definitions' not in d:
            self.__dict__.update(d)
            return

        unpickled_channels = d.pop('channel_definitions')
        for key, defs in unpickled_channels.items():
            self.channels[key] = defs
        self.__dict__.update(d)


    def __len__(self):
        return len(self.data)


    def __iter__(self):
        i = 0
        while i < len(self):
            yield self[i]
            i += 1


class Annotation(View):
    """
    An annotation is a type of View that is displayed on the top of an
    overlay. Annotations elements do not depend on the details of the
    data displayed and are generally for the convenience of the user
    (e.g. to draw attention to specific areas of the figure using
    arrows, boxes or labels).

    All annotations have an optional interval argument that indicates
    which stack elements they apply to. For instance, this allows
    annotations for a specific time interval when overlaid over a
    ViewMap or ViewMap with a 'Time' dimension. The interval
    argument is a dictionary of dimension keys and tuples containing
    (start, end) values. A value of None, indicates an unspecified
    constraint.
    """

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


    def __mul__(self, other):
        raise Exception("An annotation can only be overlaid over a different View type.")
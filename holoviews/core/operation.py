"""
ViewOperations manipulate holoviews, typically for the purposes of
analysis or visualization. Such operations apply to Views or ViewMaps
and return the appropriate objects to visualize and access the
processed data.
"""

import param

from .dimension import ViewableElement
from .element import Element, HoloMap, AxisLayout
from .layout import NdLayout
from .overlay import NdOverlay, Overlay


class ElementOperation(param.ParameterizedFunction):
    """
    A ElementOperation takes one or more elements as inputs and processes
    them, returning arbitrary new elements as output. Individual
    holoviews may be passed in directly while multiple holoviews must
    be passed in as a UniformNdMapping of the appropriate type. A
    ElementOperation may be used to implement simple dataview
    manipulations or perform complex analysis.

    Internally, ViewOperations operate on the level of individual
    holoviews, processing each layer on an input UniformNdMapping
    independently.
    """

    label = param.String(default='ElementOperation', doc="""
        The label to identify the output of the ElementOperation. By
        default this will match the name of the ElementOperation itself.""")

    def _process(self, view, key=None):
        """
        Process a single input element and output a list of views. When
        multiple views are returned as a list, they will be returned
        to the user as a NdLayout. If a UniformNdMapping is passed into a
        ElementOperation, the individual layers are processed
        sequentially and the dimension keys are passed along with
        the ViewableElement.
        """
        raise NotImplementedError


    def get_views(self, view, pattern, view_type=Element):
        """
        Helper method that return a list of views with labels ending
        with the given pattern and which have the specified type. This
        may be useful to check if a single element satisfies some
        condition or to extract the appropriate views from an NdOverlay.
        """
        if isinstance(view, (NdOverlay, Overlay)):
            matches = [v for v in view if v.label.endswith(pattern)]
        elif isinstance(view, Element):
            matches = [view] if view.label.endswith(pattern) else []

        return [match for match in matches if isinstance(match, view_type)]


    def __call__(self, view, **params):
        self.p = param.ParamOverrides(self, params)

        if isinstance(view, ViewableElement):
            views = self._process(view)
            if len(views) > 1:
                return NdLayout(views)
            else:
                return views[0]

        elif isinstance(view, AxisLayout):
            grids = []
            for pos, cell in view.items():
                val = self(cell, **params)
                maps = val.values() if isinstance(val, NdLayout) else [val]
                # Initialize the list of data or coordinate grids
                if grids == []:
                    grids = [AxisLayout(None, label=view.label) for vmap in maps]
                # Populate the grids
                for ind, vmap in enumerate(maps):
                    grids[ind][pos] = vmap

            if len(grids) == 1: return grids[0]
            else:               return NdLayout(grids)


        elif isinstance(view, HoloMap):
            mapped_items = [(k, self._process(el, key=k)) for k, el in view.items()]
            maps = [view.clone() for _ in range(len(mapped_items[0][1]))]
            for k, views in mapped_items:
                for ind, v in enumerate(views):
                    maps[ind][k] = v

            if len(maps) == 1:  return maps[0]
            else:               return NdLayout(maps)


class MapOperation(param.ParameterizedFunction):
    """
    A MapOperation takes a UniformNdMapping of Views or Overlays as inputs
    and processes them, returning arbitrary new UniformNdMapping objects as output.
    """

    label = param.String(default='MapOperation', doc="""
        The label to identifiy the output of the MapOperation. By
        default this will match the name of the MapOperation.""")


    def __call__(self, vmap, **params):
        self.p = param.ParamOverrides(self, params)

        if not isinstance(vmap, HoloMap):
            raise Exception('MapOperation can only process Maps.')

        maps = self._process(vmap)

        if len(maps) == 1:
            return maps[0]
        else:
            return NdLayout(maps)


    def _process(self, view):
        """
        Process a single input HoloMap and output a list of views or
        maps. When multiple values are returned they are returned to
        the user as a NdLayout.
        """
        raise NotImplementedError



class ChannelOperation(param.Parameterized):
    """
    A ChannelOperation defines operations to be automatically applied
    on display when certain types of Overlay are found. For instance,
    this can be used to display three overlaid monochrome matrices as
    an RGB image.

    Note that currently only operations that process Matrix elements
    are permitted.
    """

    value = param.String(doc="""
       The value identifier for the output of this particular
       ChannelOperation definition.""")

    operation = param.ClassSelector(class_=ElementOperation, is_instance=False, doc="""
       The ElementOperation to apply when combining channels""")

    pattern = param.String(doc="""
       The overlay pattern to be processed. An overlay pattern is a
       sequence of elements specified by dotted paths separated by * .

       For instance the following pattern specifies three overlayed
       matrices with values of 'RedChannel', 'GreenChannel' and
       'BlueChannel' respectively:

      'Matrix.RedChannel * Matrix.GreenChannel * Matrix.BlueChannel.

      This pattern specification could then be associated with the RGB
      operation that returns a single RGB matrix for display.""")

    kwargs = param.Dict(doc="""
       Optional set of parameters to pass to the operation.""")

    operations = []

    def __init__(self, value, pattern, operation, **kwargs):
        if not any (operation is op for op in self.operations):
            raise ValueError("Operation %r not in allowed operations" % operation)
        self._pattern_spec, labels = [], []

        for path in pattern.split('*'):
            path_tuple = tuple(el.strip() for el in path.strip().split('.'))

            if path_tuple[0] != 'Matrix':
                raise KeyError("Only Matrix is currently supported in channel operation patterns")

            self._pattern_spec.append(path_tuple)

            if len(path_tuple) == 3:
                labels.append(path_tuple[2])

        if len(labels) > 1 and not all(l==labels[0] for l in labels):
            raise KeyError("Mismatched labels not allowed in channel operation patterns")
        elif len(labels) == 1:
            self.label = labels[0]
        else:
            self.label = ''

        super(ChannelOperation, self).__init__(value=value,
                                               pattern=pattern,
                                               operation=operation,
                                               kwargs=kwargs)


    def __call__(self, overlay):
        return self.operation(overlay, label=self.label, **self.kwargs)


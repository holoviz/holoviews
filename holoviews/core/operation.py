"""
ViewOperations manipulate holoviews, typically for the purposes of
analysis or visualization. Such operations apply to Views or ViewMaps
and return the appropriate objects to visualize and access the
processed data.
"""

import param

from .dimension import ViewableElement
from .element import Element, HoloMap
from .layer import NdOverlay, AxisLayout, Overlay
from .layout import NdLayout


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

"""
ViewOperations manipulate holoviews, typically for the purposes of
analysis or visualization. Such operations apply to Views or ViewMaps
and return the appropriate objects to visualize and access the
processed data.
"""

import param

from .layer import Layer, Overlay, Grid
from .layout import GridLayout
from .view import View, Map


class ViewOperation(param.ParameterizedFunction):
    """
    A ViewOperation takes one or more views as inputs and processes
    them, returning arbitrary new view objects as output. Individual
    holoviews may be passed in directly while multiple holoviews must
    be passed in as a Map of the appropriate type. A ViewOperation
    may be used to implement simple dataview manipulations or perform
    complex analysis.

    Internally, ViewOperations operate on the level of individual
    holoviews, processing each layer on an input Map independently.
    """

    label = param.String(default='ViewOperation', doc="""
        The label to identify the output of the ViewOperation. By
        default this will match the name of the ViewOperation itself.""")

    def _process(self, view, key=None):
        """
        Process a single input view and output a list of views. When
        multiple views are returned as a list, they will be returned
        to the user as a GridLayout. If a Map is passed into a
        ViewOperation, the individual layers are processed
        sequentially and the dimension keys are passed along with
        the View.
        """
        raise NotImplementedError


    def get_views(self, view, pattern, view_type=Layer):
        """
        Helper method that return a list of views with labels ending
        with the given pattern and which have the specified type. This
        may be useful to check is a single view satisfies some
        condition or to extract the appropriate views from an Overlay.
        """
        if isinstance(view, Overlay):
            matches = [v for v in view if v.label.endswith(pattern)]
        elif isinstance(view, Layer):
            matches = [view] if view.label.endswith(pattern) else []

        return [match for match in matches if isinstance(match, view_type)]


    def __call__(self, view, **params):
        self.p = param.ParamOverrides(self, params)

        if isinstance(view, View):
            views = self._process(view)
            if len(views) > 1:
                return GridLayout(views)
            else:
                return views[0]

        elif isinstance(view, Grid):
            grids = []
            for pos, cell in view.items():
                val = self(cell, **params)
                stacks = val.values() if isinstance(val, GridLayout) else [val]
                # Initialize the list of data or coordinate grids
                if grids == []:
                    grids = [Grid(view.bounds, None, view.xdensity, view.ydensity, label=view.label)
                             for stack in stacks]
                # Populate the grids
                for ind, stack in enumerate(stacks):
                    grids[ind][pos] = stack

            if len(grids) == 1: return grids[0]
            else:               return GridLayout(grids)


        elif isinstance(view, Map):
            mapped_items = [(k, self._process(el, key=k)) for k, el in view.items()]
            stacks = [view.clone(dimensions=view.dimensions) for _ in range(len(mapped_items[0][1]))]
            for k, views in mapped_items:
                for ind, v in enumerate(views):
                    stacks[ind][k] = v

            if len(stacks) == 1:  return stacks[0]
            else:                 return GridLayout(stacks)


class MapOperation(param.ParameterizedFunction):
    """
    A MapOperation takes a Map of Views or Overlays as inputs
    and processes them, returning arbitrary new Map objects as output.
    """

    label = param.String(default='MapOperation', doc="""
        The label to identifiy the output of the MapOperation. By
        default this will match the name of the MapOperation.""")


    def __call__(self, stack, **params):
        self.p = param.ParamOverrides(self, params)

        if not isinstance(stack, Map):
            raise Exception('MapOperation can only process Maps.')

        stacks = self._process(stack)

        if len(stacks) == 1:
            return stacks[0]
        else:
            return GridLayout(stacks)


    def _process(self, view):
        """
        Process a single input Map and output a list of views or
        stacks. When multiple values are returned they are returned to
        the user as a GridLayout.
        """
        raise NotImplementedError

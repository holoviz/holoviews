"""
Operations manipulate Elements, HoloMaps and Layouts, typically for
the purposes of analysis or visualization.
"""

import numpy as np

import param

from .dimension import ViewableElement
from .element import Element, HoloMap, AxisLayout
from .layout import NdLayout, LayoutTree
from .overlay import CompositeOverlay, NdOverlay, Overlay


class Operation(param.ParameterizedFunction):
    """
    Base class for all Operation types.
    """

    value = param.String(default='Operation', doc="""
       The value string used to identify the output of the
       Operation. By default this should match the operation name.""")


    @classmethod
    def search(cls, element, pattern):
        """
        Helper method that returns a list of elements that match the
        given path pattern of form {type}.{value}.{label}.

        The input may be a LayoutTree, an Overlay type or a single
        Element.
        """
        if isinstance(element, LayoutTree):
            return [el for cell in element for el in cls.search(cell, pattern)]
        if isinstance(element, (NdOverlay, Overlay)):
            return [el for el in element if el.matches(pattern)]
        elif isinstance(element, Element):
            return [element] if element.matches(pattern) else []


    @classmethod
    def get_overlay_label(cls, overlay, default_label=''):
        """
        Returns a label if all the elements of an overlay agree on a
        consistent label, otherwise returns the default label.
        """
        if all(el.label==overlay[0].label for el in overlay):
            return overlay[0].label
        else:
            return default_label


class ElementOperation(Operation):
    """
    An ElementOperation process an Element or HoloMap at the level of
    individual elements or overlays. If a holomap is passed in as
    input, a processed holomap is returned as output where the
    individual elements have been transformed accordingly. An
    ElementOperation may turn overlays in new elements or vice versa.
    """

    def _process(self, view, key=None):
        """
        Process a single input element and outputs new single element
        or overlay. If a HoloMap is passed into a ElementOperation,
        the individual components are processed sequentially with the
        corresponding key passed as the optional key argument.
        """
        raise NotImplementedError


    def __call__(self, element, **params):
        self.p = param.ParamOverrides(self, params)

        if isinstance(element, ViewableElement):
            processed = self._process(element)
        elif isinstance(element, AxisLayout):
            # Initialize an empty axis layout
            processed = AxisLayout(None, label=element.label)
            # Populate the axis layout
            for pos, cell in element.items():
                processed[pos] = self(cell, **params)
        elif isinstance(element, HoloMap):
            mapped_items = [(k, self._process(el, key=k))
                            for k, el in element.items()]
            processed = element.clone(mapped_items)
        return processed



class MapOperation(param.ParameterizedFunction):
    """
    A MapOperation takes a HoloMap containing elements or overlays and
    processes them at the HoloMap level, returning arbitrary new
    HoloMap objects as output. Unlike ElementOperation, MapOperations
    can compute over all the keys and dimensions of the input map.
    """

    value = param.String(default='MapOperation', doc="""
        The value string to identify the output of the MapOperation.
        By default this will match the MapOperation name.""")

    def __call__(self, vmap, **params):
        self.p = param.ParamOverrides(self, params)

        if not isinstance(vmap, HoloMap):
            raise Exception('MapOperation can only process Maps.')

        return self._process(vmap)


    def _process(self, view):
        """
        Process a single input HoloMap, returning a new HoloMap
        instance.
        """
        raise NotImplementedError



class LayoutOperation(param.ParameterizedFunction):

    pass



class ChannelOperation(param.Parameterized):
    """
    A ChannelOperation defines operations to be automatically applied
    on display when certain types of Overlay are found. For instance,
    this can be used to display three overlaid monochrome matrices as
    an RGB image.

    Note that currently only operations that process Matrix elements
    are permitted.
    """

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

    value = param.String(doc="""
       The value identifier for the output of this particular
       ChannelOperation definition.""")

    kwargs = param.Dict(doc="""
       Optional set of parameters to pass to the operation.""")

    operations = []

    channel_ops = []

    def __init__(self, pattern, operation, value, **kwargs):
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


    @classmethod
    def _collapse(cls, overlay, pattern, fn, style_key):
        """
        Given an overlay object collapse the channels according to
        pattern using the supplied function. Any collapsed ViewableElement is
        then given the supplied style key.
        """
        pattern = [el.strip() for el in pattern.rsplit('*')]
        if len(pattern) > len(overlay): return overlay

        skip=0
        collapsed_overlay = overlay.clone(None)
        for i, key in enumerate(overlay.keys()):
            layer_labels = overlay.labels[i:len(pattern)+i]
            matching = all(l.endswith(p) for l, p in zip(layer_labels, pattern))
            if matching and len(layer_labels)==len(pattern):
                views = [el for el in overlay if el.label in layer_labels]
                if isinstance(overlay, Overlay):
                    views = np.product([Overlay.from_view(el) for el in overlay])
                overlay_slice = overlay.clone(views)
                collapsed_view = fn(overlay_slice)
                if isinstance(overlay, LayoutTree):
                    collapsed_overlay *= collapsed_view
                else:
                    collapsed_overlay[key] = collapsed_view
                skip = len(views)-1
            elif skip:
                skip = 0 if skip <= 0 else (skip - 1)
            else:
                if isinstance(overlay, LayoutTree):
                    collapsed_overlay *= overlay[key]
                else:
                    collapsed_overlay[key] = overlay[key]
        return collapsed_overlay


    @classmethod
    def collapse_channels(cls, vmap):
        """
        Given a map of Overlays, apply all applicable channel
        reductions.
        """
        if not issubclass(vmap.type, CompositeOverlay):
            return vmap
        elif not CompositeOverlay.channels.keys(): # No potential channel reductions
            return vmap

        # Apply all customized channel operations
        collapsed_vmap = vmap.clone()
        for key, overlay in vmap.items():
            customized = [k for k in CompositeOverlay.channels.keys()
                          if overlay.label and k.startswith(overlay.label)]
            # Largest reductions should be applied first
            sorted_customized = sorted(customized, key=lambda k: -CompositeOverlay.channels[k].size)
            sorted_reductions = sorted(CompositeOverlay.channels.options(),
                                       key=lambda k: -CompositeOverlay.channels[k].size)
            # Collapse the customized channel before the other definitions
            for key in sorted_customized + sorted_reductions:
                channel = CompositeOverlay.channels[key]
                if channel.mode is None: continue
                collapse_fn = channel.operation
                fn = collapse_fn.instance(**channel.opts)
                collapsed_vmap[k] = cls._collapse(overlay, channel.pattern, fn, key)
        return vmap

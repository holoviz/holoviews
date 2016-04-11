"""
Operations manipulate Elements, HoloMaps and Layouts, typically for
the purposes of analysis or visualization.
"""
from functools import reduce
import param

try:
    from itertools import izip as zip
except:
    pass

from .dimension import ViewableElement
from .element import Element, HoloMap, GridSpace, Collator
from .layout import Layout
from .overlay import NdOverlay, Overlay
from .spaces import DynamicMap
from .traversal import unique_dimkeys
from . import util


class Operation(param.ParameterizedFunction):
    """
    Base class for all Operation types.
    """

    group = param.String(default='Operation', doc="""
       The group string used to identify the output of the
       Operation. By default this should match the operation name.""")


    @classmethod
    def search(cls, element, pattern):
        """
        Helper method that returns a list of elements that match the
        given path pattern of form {type}.{group}.{label}.

        The input may be a Layout, an Overlay type or a single
        Element.
        """
        if isinstance(element, Layout):
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
        if all(el.label==overlay.get(0).label for el in overlay):
            return overlay.get(0).label
        else:
            return default_label


    @classmethod
    def get_overlay_bounds(cls, overlay):
        """
        Returns the extents if all the elements of an overlay agree on
        a consistent extents, otherwise raises an exception.
        """
        if all(el.bounds==overlay.get(0).bounds for el in overlay):
            return overlay.get(0).bounds
        else:
            raise ValueError("Extents across the overlay are inconsistent")


class DynamicOperation(Operation):
    """
    Dynamically applies an operation to the elements of a HoloMap
    or DynamicMap. Will return a DynamicMap wrapping the original
    map object, which will lazily evaluate when a key is requested.
    The _process method should be overridden in subclasses to apply
    a specific operation, DynamicOperation itself applies a no-op,
    making the DynamicOperation baseclass useful for converting
    existing HoloMaps to a DynamicMap.
    """

    def __call__(self, map_obj, **params):
        self.p = param.ParamOverrides(self, params)
        callback = self._dynamic_operation(map_obj)
        if isinstance(map_obj, DynamicMap):
            return map_obj.clone(callback=callback, shared_data=False)
        else:
            return self._make_dynamic(map_obj, callback)


    def _process(self, element):
        return element


    def _dynamic_operation(self, map_obj):
        """
        Generate function to dynamically apply the operation.
        Wraps an existing HoloMap or DynamicMap.
        """
        if not isinstance(map_obj, DynamicMap):
            def dynamic_operation(*key):
                return self._process(map_obj[key])
            return dynamic_operation

        def dynamic_operation(*key):
            key = key[0] if map_obj.mode == 'open' else key
            _, el = util.get_dynamic_item(map_obj, map_obj.kdims, key)
            return self._process(el)

        return dynamic_operation


    def _make_dynamic(self, hmap, dynamic_fn):
        """
        Accepts a HoloMap and a dynamic callback function creating
        an equivalent DynamicMap from the HoloMap.
        """
        dim_values = zip(*hmap.data.keys())
        params = util.get_param_values(hmap)
        kdims = [d(values=list(values)) for d, values in zip(hmap.kdims, dim_values)]
        return DynamicMap(dynamic_fn, **dict(params, kdims=kdims))



class DynamicFunction(DynamicOperation):
    """
    Dynamically applies a function to the Elements in a DynamicMap
    or HoloMap. Must supply a HoloMap or DynamicMap type and will
    return another DynamicMap type, which will apply the supplied
    function with the supplied kwargs whenever a value is requested
    from the map.
    """

    function = param.Callable(default=lambda x: x, doc="""
        Function to apply to DynamicMap items dynamically.""")

    kwargs = param.Dict(default={}, doc="""
        Keyword arguments passed to the function.""")

    def _process(self, element):
        return self.p.function(element, **self.p.kwargs)



class ElementOperation(Operation):
    """
    An ElementOperation process an Element or HoloMap at the level of
    individual elements or overlays. If a holomap is passed in as
    input, a processed holomap is returned as output where the
    individual elements have been transformed accordingly. An
    ElementOperation may turn overlays in new elements or vice versa.
    """

    dynamic = param.ObjectSelector(default='default',
                                   objects=['default', True, False], doc="""
       Whether the operation should be applied dynamically when a
       specific frame is requested, specified as a Boolean. If set to
       'default' the mode will be determined based on the input type,
       i.e. if the data is a DynamicMap it will stay dynamic.""")

    input_ranges = param.ClassSelector(default={},
                                       class_=(dict, tuple), doc="""
       Ranges to be used for input normalization (if applicable) in a
       format appropriate for the Normalization.ranges parameter.

       By default, no normalization is applied. If key-wise
       normalization is required, a 2-tuple may be supplied where the
       first component is a Normalization.ranges list and the second
       component is Normalization.keys. """)


    def _process(self, view, key=None):
        """
        Process a single input element and outputs new single element
        or overlay. If a HoloMap is passed into a ElementOperation,
        the individual components are processed sequentially with the
        corresponding key passed as the optional key argument.
        """
        raise NotImplementedError


    def process_element(self, element, key, **params):
        """
        The process_element method allows a single element to be
        operated on given an externally supplied key.
        """
        self.p = param.ParamOverrides(self, params)
        return self._process(element, key)


    def __call__(self, element, **params):
        self.p = param.ParamOverrides(self, params)
        dynamic = ((self.p.dynamic == 'default' and
                    isinstance(element, DynamicMap))
                   or self.p.dynamic is True)

        if isinstance(element, ViewableElement):
            processed = self._process(element)
        elif isinstance(element, GridSpace):
            # Initialize an empty axis layout
            grid_data = ((pos, self(cell, **params))
                         for pos, cell in element.items())
            processed = GridSpace(grid_data, label=element.label,
                                  kdims=element.kdims)
        elif dynamic:
            processed = DynamicFunction(element, function=self, kwargs=params)
        elif isinstance(element, DynamicMap):
            if any((not d.values) for d in element.kdims):
                raise ValueError('Applying a non-dynamic operation requires '
                                 'all DynamicMap key dimensions to define '
                                 'the sampling by specifying values.')
            samples = tuple(d.values for d in element.kdims)
            processed = self(element[samples], **params)
        elif isinstance(element, HoloMap):
            mapped_items = [(k, self._process(el, key=k))
                            for k, el in element.items()]
            refval = mapped_items[0][1]
            processed = element.clone(mapped_items,
                                      group=refval.group,
                                      label=refval.label)
        else:
            raise ValueError("Cannot process type %r" % type(element).__name__)
        return processed



class MapOperation(param.ParameterizedFunction):
    """
    A MapOperation takes a HoloMap containing elements or overlays and
    processes them at the HoloMap level, returning arbitrary new
    HoloMap objects as output. Unlike ElementOperation, MapOperations
    can compute over all the keys and dimensions of the input map.
    """

    group = param.String(default='MapOperation', doc="""
        The group string to identify the output of the MapOperation.
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



class TreeOperation(Operation):
    """
    A TreeOperation is the most general Operation type; it accepts any
    HoloViews datastructure and outputs a Layout containing one or
    more elements.
    """

    def process_element(self, element, key, **params):
        """
        The process_element method allows a single element to be
        operated on given an externally supplied key.
        """
        self.p = param.ParamOverrides(self, params)
        maps = self._process(element, key)
        return reduce(lambda x,y: x + y, maps)


    def __call__(self, src, **params):
        self.p = param.ParamOverrides(self, params)
        dims, keys = unique_dimkeys(src)
        if isinstance(src, Layout) and not src.uniform:
            raise Exception("TreeOperation can only process uniform Layouts")

        if not dims:
            return self.process_element(src, None)
        else:
            dim_names = [d.name for d in dims]
            values = {}
            for key in keys:
                selection = src.select(**dict(zip(dim_names, key)))
                if not isinstance(selection, Layout):
                    selection = Layout.from_values([selection])
                processed = self._process(selection, key)
                if isinstance(processed, list):
                    processed = Layout.from_values(processed)
                values[key] = processed
        return Collator(values, kdims=dims)()



    def _process(self, tree, key=None):
        """
        Process a single input Layout, returning a list of
        elements to be merged with the output Layout.
        """
        raise NotImplementedError


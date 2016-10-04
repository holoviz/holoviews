import param

from .core import DynamicMap, ViewableElement
from .core.operation import ElementOperation


class Dynamic(param.ParameterizedFunction):
    """
    Dynamically applies a callable to the Elements in any HoloViews
    object. Will return a DynamicMap wrapping the original map object,
    which will lazily evaluate when a key is requested. By default
    Dynamic applies a no-op, making it useful for converting HoloMaps
    to a DynamicMap.

    Any supplied kwargs will be passed to the callable and any streams
    will be instantiated on the returned DynamicMap.
    """

    operation = param.Callable(default=lambda x: x, doc="""
        Callable to apply to DynamicMap items dynamically.""")

    kwargs = param.Dict(default={}, doc="""
        Keyword arguments passed to the function.""")

    streams = param.List(default=[], doc="""
        List of streams to attach to the returned DynamicMap""")

    def __call__(self, map_obj, **params):
        self.p = param.ParamOverrides(self, params)
        callback = self._dynamic_operation(map_obj)
        if isinstance(map_obj, DynamicMap):
            dmap = map_obj.clone(callback=callback, shared_data=False)
        else:
            dmap = self._make_dynamic(map_obj, callback)
        if isinstance(self.p.operation, ElementOperation):
            return dmap.clone(streams=[s() for s in self.p.streams])
        return dmap


    def _process(self, element, key=None):
        if isinstance(self.p.operation, ElementOperation):
            return self.p.operation.process_element(element, key, **self.p.kwargs)
        else:
            return self.p.operation(element, **self.p.kwargs)


    def _dynamic_operation(self, map_obj):
        """
        Generate function to dynamically apply the operation.
        Wraps an existing HoloMap or DynamicMap.
        """
        if not isinstance(map_obj, DynamicMap):
            def dynamic_operation(*key, **kwargs):
                self.p.kwargs.update(kwargs)
                return self._process(map_obj[key], key)
            return dynamic_operation

        def dynamic_operation(*key, **kwargs):
            key = key[0] if map_obj.mode == 'open' else key
            self.p.kwargs.update(kwargs)
            _, el = util.get_dynamic_item(map_obj, map_obj.kdims, key)
            return self._process(el, key)

        return dynamic_operation


    def _make_dynamic(self, hmap, dynamic_fn):
        """
        Accepts a HoloMap and a dynamic callback function creating
        an equivalent DynamicMap from the HoloMap.
        """
        if isinstance(hmap, ViewableElement):
            return DynamicMap(dynamic_fn, kdims=[])
        dim_values = zip(*hmap.data.keys())
        params = util.get_param_values(hmap)
        kdims = [d(values=list(set(values))) for d, values in
                 zip(hmap.kdims, dim_values)]
        return DynamicMap(dynamic_fn, **dict(params, kdims=kdims))

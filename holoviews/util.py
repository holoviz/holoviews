import inspect

import param

from .core import DynamicMap, ViewableElement
from .core.operation import ElementOperation
from .core.util import Aliases
from .core.operation import OperationCallable
from .core.spaces import Callable
from .core import util
from .streams import Stream

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
        Operation or user-defined callable to apply dynamically""")

    kwargs = param.Dict(default={}, doc="""
        Keyword arguments passed to the function.""")

    link_inputs = param.Boolean(default=True, doc="""
         Whether a plot created from the output of this utility should
         be linked to the inputs passed to this utility.""")

    shared_data = param.Boolean(default=False, doc="""
        Whether the cloned DynamicMap will share the same cache.""")

    streams = param.List(default=[], doc="""
        List of streams to attach to the returned DynamicMap""")

    def __call__(self, map_obj, **params):
        self.p = param.ParamOverrides(self, params)
        callback = self._dynamic_operation(map_obj)
        if isinstance(map_obj, DynamicMap):
            dmap = map_obj.clone(callback=callback, shared_data=self.p.shared_data,
                                 streams=[])
        else:
            dmap = self._make_dynamic(map_obj, callback)
        if isinstance(self.p.operation, ElementOperation):
            streams = []
            for stream in self.p.streams:
                if inspect.isclass(stream) and issubclass(stream, Stream):
                    stream = stream()
                elif not isinstance(stream, Stream):
                    raise ValueError('Stream must only contain Stream '
                                     'classes or instances')
                updates = {k: self.p.operation.p.get(k) for k, v in stream.contents.items()
                           if v is None and k in self.p.operation.p}
                if updates:
                    stream.update(trigger=False, **updates)
                streams.append(stream)
            return dmap.clone(streams=streams)
        return dmap


    def _process(self, element, key=None):
        if isinstance(self.p.operation, ElementOperation):
            kwargs = {k: v for k, v in self.p.kwargs.items()
                      if k in self.p.operation.params()}
            return self.p.operation.process_element(element, key, **kwargs)
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
        else:
            def dynamic_operation(*key, **kwargs):
                self.p.kwargs.update(kwargs)
                safe_key = () if not map_obj.kdims else key
                el = map_obj[key]
                return self._process(el, key)
        if isinstance(self.p.operation, ElementOperation):
            return OperationCallable(dynamic_operation, inputs=[map_obj],
                                     link_inputs=self.p.link_inputs,
                                     operation=self.p.operation)
        else:
            return Callable(dynamic_operation, inputs=[map_obj],
                            link_inputs=self.p.link_inputs)


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

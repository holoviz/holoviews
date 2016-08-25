import param
import uuid
from collections import OrderedDict


class Stream(param.Parameterized):

    # Mapping from uuid to stream instance
    registry = OrderedDict()

    @classmethod
    def trigger(cls, streams):
        pass

    def update(self, params):
        self.set_param(**params)
        for callback in self.callbacks + self._hidden_callbacks:
            callback(params)

    @classmethod
    def find(cls, obj):
        """
        Return a set of streams from the registry with a given source.
        """
        return set(v for v in cls.registry.values() if v.source is obj)

    def __init__(self, mapping=None, source=None, callbacks=[], **params):
        """
        Mapping allows multiple streams with similar event state to be
        used by remapping parameter names.

        Source is an optional argument specifying the HoloViews
        datastructure that the stream receives events from, as supported
        by the plotting backend.
        """
        self.source = source
        self.callbacks = callbacks
        self._hidden_callbacks = []

        if mapping is None:
            self.mapping = {}
        elif isinstance(mapping, dict):
            # Could do some validation here
            self.mapping = mapping
        elif len(self.params()) == 2:
            self.mapping = {k:mapping for k in self.params() if k != 'name'}
        else:
            raise Exception("Stream has multiple parameters, please supply a dictionary")

        self.uuid = uuid.uuid4().hex
        super(Stream, self).__init__(**params)
        self.registry[self.uuid] = self

    @property
    def value(self):
        return {self.mapping.get(k,k):v for (k,v) in self.get_param_values()
                if k != 'name'}

    def __repr__(self):
        cls_name = self.__class__.__name__
        kwargs = ','.join('%s=%r' % (k,v)
                          for (k,v) in self.get_param_values() if k != 'name')
        if not self.mapping:
            return '%s(%s)' % (cls_name, kwargs)
        elif len(self.mapping) == 1:
            return '%s(%r, %s)' % (cls_name, self.mapping.values()[0], kwargs)
        else:
            return '%s(%r, %s)' % (cls_name, self.mapping, kwargs)

    def __str__(self):
        return repr(self)


class MouseX(Stream):

    x = param.Number(default=0)

    def __init__(self, mapping=None, **params):
        super(MouseX, self).__init__(mapping=mapping, **params)


class MouseY(Stream):

    y = param.Number(default=0)

    def __init__(self, mapping=None, **params):
        super(MouseY, self).__init__(mapping=mapping, **params)


class MouseXY(Stream):

    x = param.Number(default=0)

    y = param.Number(default=0)

    def __init__(self, mapping=None, **params):
        super(MouseXY, self).__init__(mapping=mapping, **params)

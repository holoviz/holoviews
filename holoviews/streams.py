"""
The streams module defines the streams API that allows visualizations to
generate and respond to events, originating either in Python on the
server-side or in Javascript in the Jupyter notebook (client-side).
"""

import param
import uuid
from collections import OrderedDict


class Stream(param.Parameterized):
    """
    A Stream is simply a parameterized object with parameters that
    change over time in response to update events. Parameters are
    updated via the update method.

    Streams may have one or more subscribers, callables that are passed
    the parameter dictionary when the trigger classmethod is called.
    """

    # Mapping from uuid to stream instance
    registry = OrderedDict()

    @classmethod
    def trigger(cls, streams):
        pass

    def update(self, params):
        self.set_param(**params)
        for subscriber in self.subscribers + self._hidden_subscribers:
            subscriber(params)

    @classmethod
    def find(cls, obj):
        """
        Return a set of streams from the registry with a given source.
        """
        return set(v for v in cls.registry.values() if v.source is obj)

    def __init__(self, mapping=None, source=None, subscribers=[], **params):
        """
        Mapping allows multiple streams with similar event state to be
        used by remapping parameter names.

        Source is an optional argument specifying the HoloViews
        datastructure that the stream receives events from, as supported
        by the plotting backend.
        """
        self.source = source
        self.subscribers = subscribers
        self._hidden_subscribers = []

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

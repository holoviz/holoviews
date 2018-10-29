import weakref
from collections import defaultdict

import param


class Link(param.Parameterized):
    """
    A Link defines some connection between a source and target object
    in their visualization. It is quite similar to a Stream as it
    allows defining callbacks in response to some change or event on
    the source object, however, unlike a Stream, it does not transfer
    data and make it available to user defined subscribers. Instead
    a Link directly causes some action to occur on the target, for JS
    based backends this usually means that a corresponding JS callback
    will effect some change on the target in response to a change on
    the source.

    A Link must define a source object which is what triggers events,
    but must not define a target. It is also possible to define bi-
    directional links between the source and target object.
    """

    # Mapping from a source id to a Link instance
    registry = weakref.WeakKeyDictionary()

    # Mapping to define callbacks by backend and Link type.
    # e.g. Link._callbacks['bokeh'][Stream] = Callback
    _callbacks = defaultdict(dict)

    def __init__(self, source, target=None, **params):
        if source is None:
            raise ValueError('%s must define a source' % type(self).__name__)

        # Source is stored as a weakref to allow it to be garbage collected
        self._source = weakref.ref(source) if source else None
        self._target = weakref.ref(target) if target else None
        super(Link, self).__init__(**params)
        self.link()

    @classmethod
    def register_callback(cls, backend, callback):
        """
        Register a LinkCallback providing the implementation for
        the Link for a particular backend.
        """
        cls._callbacks[backend][cls] = callback

    @property
    def source(self):
        return self._source() if self._source else None

    @property
    def target(self):
        return self._target() if self._target else None

    def link(self):
        """
        Registers the Link
        """
        if self.source in self.registry:
            self.registry[self.source].append(self)
        else:
            self.registry[self.source] = [self]

    def unlink(self):
        """
        Unregisters the Link
        """
        links = self.registry.get(self.source)
        if self in links:
            links.pop(links.index(self))


class RangeToolLink(Link):
    """
    The RangeToolLink sets up a link between a RangeTool on the source
    plot and the axes on the target plot. It is useful for exploring
    a subset of a larger dataset in more detail. By default it will
    link along the x-axis but using the axes parameter both axes may
    be linked to the tool.
    """

    axes = param.ListSelector(default=['x'], objects=['x', 'y'], doc="""
        Which axes to link the tool to.""")


class DataLink(Link):
    """
    DataLink defines a link in the data between two objects allowing
    them to be selected together. In order for a DataLink to be
    established the source and target data must be of the same length.
    """
 
    def __init__(self, source, target, **params):
        if source is None or target is None:
            raise ValueError('%s must define a source and a target.')
        super(DataLink, self).__init__(source=source, target=target, **params)

from collections import defaultdict

import param

from ..core import ViewableElement


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

    source = param.ClassSelector(class_=ViewableElement, doc="""
        The source object of the link (required).""")

    target = param.ClassSelector(class_=ViewableElement, doc="""
        The target object of the link (optional).""")

    # Mapping from a source id to a Link instance
    registry = {}

    # Mapping to define callbacks by backend and Link type.
    # e.g. Link._callbacks['bokeh'][Stream] = Callback
    _callbacks = defaultdict(dict)

    def __init__(self, source, target=None, **params):
        if source is None:
            raise ValueError('%s must define a source' % type(self).__name__)
        super(Link, self).__init__(source=source, target=target, **params)
        self.registry[id(source)] = self


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

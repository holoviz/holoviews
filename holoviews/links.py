from collections import defaultdict

import param

from .core import ViewableElement
from .element import Path, Table


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
    """

    source = param.ClassSelector(class_=ViewableElement)

    target = param.ClassSelector(class_=ViewableElement)

    # Mapping from a source id to a Link instance
    registry = {}

    # Mapping to define callbacks by backend and Link type.
    # e.g. Link._callbacks['bokeh'][Stream] = Callback
    _callbacks = defaultdict(dict)

    def __init__(self, source, target):
        super(Link, self).__init__(source=source, target=target)
        self.registry[id(source)] = self


class PathTableLink(Link):
    """
    Links the currently selected Path to a Table.
    """

    source = param.ClassSelector(class_=Path)

    target = param.ClassSelector(class_=Table)

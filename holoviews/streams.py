"""
The streams module defines the streams API that allows visualizations to
generate and respond to events, originating either in Python on the
server-side or in Javascript in the Jupyter notebook (client-side).
"""

import param
import numpy as np
from numbers import Number
from collections import defaultdict
from .core import util

from contextlib import contextmanager


@contextmanager
def triggering_streams(streams):
    """
    Temporarily declares the streams as being in a triggered state.
    Needed by DynamicMap to determine whether to memoize on a Callable,
    i.e. if a stream has memoization disabled and is in triggered state
    Callable should disable lookup in the memoization cache. This is
    done by the dynamicmap_memoization context manager.
    """
    for stream in streams:
        stream._triggering = True
    try:
        yield
    except:
        raise
    finally:
        for stream in streams:
            stream._triggering = False


class Stream(param.Parameterized):
    """
    A Stream is simply a parameterized object with parameters that
    change over time in response to update events and may trigger
    downstream events on its subscribers. The Stream parameters can be
    updated using the update method, which will optionally trigger the
    stream. This will notify the subscribers which may be supplied as
    a list of callables or added later using the add_subscriber
    method. The subscribers will be passed a dictionary mapping of the
    parameters of the stream, which are available on the instance as
    the ``contents``.

    Depending on the plotting backend certain streams may
    interactively subscribe to events and changes by the plotting
    backend. For this purpose use the LinkedStream baseclass, which
    enables the linked option by default. A source for the linking may
    be supplied to the constructor in the form of another viewable
    object specifying which part of a plot the data should come from.

    The transient option allows treating stream events as discrete
    updates, resetting the parameters to their default after the
    stream has been triggered. A downstream callback can therefore
    determine whether a stream is active by checking whether the
    stream values match the default (usually None).

    The Stream class is meant for subclassing and subclasses should
    generally add one or more parameters but may also override the
    transform and reset method to preprocess parameters before they
    are passed to subscribers and reset them using custom logic
    respectively.
    """

    # Mapping from a source id to a list of streams
    registry = defaultdict(list)

    # Mapping to define callbacks by backend and Stream type.
    # e.g. Stream._callbacks['bokeh'][Stream] = Callback
    _callbacks = defaultdict(dict)


    @classmethod
    def define(cls, name, **kwargs):
        """
        Utility to quickly and easily declare Stream classes. Designed
        for interactive use such as notebooks and shouldn't replace
        parameterized class definitions in source code that is imported.

        Takes a stream class name and a set of keywords where each
        keyword becomes a parameter. If the value is already a
        parameter, it is simply used otherwise the appropriate parameter
        type is inferred and declared, using the value as the default.

        Supported types: bool, int, float, str, dict, tuple and list
        """
        params = {'name':param.String(default=name)}
        for k,v in kwargs.items():
            kws = dict(default=v, constant=True)
            if isinstance(v, param.Parameter):
                params[k] = v
            elif isinstance(v, bool):
                params[k] = param.Boolean(**kws)
            elif isinstance(v, int):
                params[k] = param.Integer(**kws)
            elif isinstance(v, float):
                params[k] = param.Number(**kws)
            elif isinstance(v,str):
                params[k] = param.String(**kws)
            elif isinstance(v,dict):
                params[k] = param.Dict(**kws)
            elif isinstance(v, tuple):
                params[k] = param.Tuple(**kws)
            elif isinstance(v,list):
                params[k] = param.List(**kws)
            elif isinstance(v,np.ndarray):
                params[k] = param.Array(**kws)
            else:
                params[k] = param.Parameter(**kws)

        # Dynamic class creation using type
        return type(name, (Stream,), params)


    @classmethod
    def trigger(cls, streams):
        """
        Given a list of streams, collect all the stream parameters into
        a dictionary and pass it to the union set of subscribers.

        Passing multiple streams at once to trigger can be useful when a
        subscriber may be set multiple times across streams but only
        needs to be called once.
        """
        # Union of stream contents
        items = [stream.contents.items() for stream in streams]
        union = [kv for kvs in items for kv in kvs]
        klist = [k for k,_ in union]
        clashes = set([k for k in klist if klist.count(k) > 1])
        if clashes:
            param.main.warning('Parameter name clashes for keys: %r' % clashes)

        # Group subscribers by precedence while keeping the ordering
        # within each group
        subscriber_precedence = defaultdict(list)
        for stream in streams:
            for precedence, subscriber in stream._subscribers:
                subscriber_precedence[precedence].append(subscriber)
        sorted_subscribers = sorted(subscriber_precedence.items(), key=lambda x: x[0])
        subscribers = util.unique_iterator([s for _, subscribers in sorted_subscribers
                                            for s in subscribers])

        with triggering_streams(streams):
            for subscriber in subscribers:
                subscriber(**dict(union))

        for stream in streams:
            with util.disable_constant(stream):
                if stream.transient:
                    stream.reset()


    def __init__(self, rename={}, source=None, subscribers=[], linked=False,
                 transient=False, **params):
        """
        The rename argument allows multiple streams with similar event
        state to be used by remapping parameter names.

        Source is an optional argument specifying the HoloViews
        datastructure that the stream receives events from, as supported
        by the plotting backend.

        Some streams are configured to automatically link to the source
        plot, to disable this set linked=False
        """
        self._source = source
        self._subscribers = []
        for subscriber in subscribers:
            self.add_subscriber(subscriber)

        self.linked = linked
        self._rename = self._validate_rename(rename)
        self.transient = transient

        # Whether this stream is currently triggering its subscribers
        self._triggering = False

        # The metadata may provide information about the currently
        # active event, i.e. the source of the stream values may
        # indicate where the event originated from
        self._metadata = {}

        super(Stream, self).__init__(**params)
        if source:
            self.registry[id(source)].append(self)


    @property
    def subscribers(self):
        " Property returning the subscriber list"
        return [s for p, s in sorted(self._subscribers)]


    def clear(self, policy='all'):
        """
        Clear all subscribers registered to this stream.

        The default policy of 'all' clears all subscribers. If policy is
        set to 'user', only subscribers defined by the user are cleared
        (precedence between zero and one). A policy of 'internal' clears
        subscribers with precedence greater than unity used internally
        by HoloViews.
        """
        policies = ['all', 'user', 'internal']
        if policy not in policies:
            raise ValueError('Policy for clearing subscribers must be one of %s' % policies)
        if policy == 'all':
            remaining = []
        elif policy == 'user':
            remaining = [(p,s) for (p,s) in self._subscribers if p > 1]
        else:
            remaining = [(p,s) for (p,s) in self._subscribers if p <= 1]
        self._subscribers = remaining


    def reset(self):
        """
        Resets stream parameters to their defaults.
        """
        with util.disable_constant(self):
            for k, p in self.params().items():
                if k != 'name':
                    setattr(self, k, p.default)


    def add_subscriber(self, subscriber, precedence=0):
        """
        Register a callable subscriber to this stream which will be
        invoked either when event is called or when this stream is
        passed to the trigger classmethod.

        Precedence allows the subscriber ordering to be
        controlled. Users should only add subscribers with precedence
        between zero and one while HoloViews itself reserves the use of
        higher precedence values. Subscribers with high precedence are
        invoked later than ones with low precedence.
        """
        if not callable(subscriber):
            raise TypeError('Subscriber must be a callable.')
        self._subscribers.append((precedence, subscriber))


    def _validate_rename(self, mapping):
        param_names = [k for k in self.params().keys() if k != 'name']
        for k,v in mapping.items():
            if k not in param_names:
                raise KeyError('Cannot rename %r as it is not a stream parameter' % k)
            if v in param_names:
                raise KeyError('Cannot rename to %r as it clashes with a '
                               'stream parameter of the same name' % v)
        return mapping


    def rename(self, **mapping):
        """
        The rename method allows stream parameters to be allocated to
        new names to avoid clashes with other stream parameters of the
        same name. Returns a new clone of the stream instance with the
        specified name mapping.
        """
        params = {k:v for k,v in self.get_param_values() if k != 'name'}
        return self.__class__(rename=mapping,
                              source=self._source,
                              linked=self.linked, **params)

    @property
    def source(self):
        return self._source


    @source.setter
    def source(self, source):
        if self._source:
            source_list = self.registry[id(self._source)]
            if self in source_list:
                source_list.remove(self)
        self._source = source
        self.registry[id(source)].append(self)


    def transform(self):
        """
        Method that can be overwritten by subclasses to process the
        parameter values before renaming is applied. Returns a
        dictionary of transformed parameters.
        """
        return {}


    @property
    def contents(self):
        filtered = {k:v for k,v in self.get_param_values() if k!= 'name' }
        return {self._rename.get(k,k):v for (k,v) in filtered.items()}


    def _set_stream_parameters(self, **kwargs):
        """
        Sets the stream parameters which are expected to be declared
        constant.
        """
        with util.disable_constant(self):
            self.set_param(**kwargs)

    def event(self, **kwargs):
        """
        Update the stream parameters and trigger an event.
        """
        self.update(**kwargs)
        self.trigger([self])

    def update(self, **kwargs):
        """
        The update method updates the stream parameters (without any
        renaming applied) in response to some event. If the stream has a
        custom transform method, this is applied to transform the
        parameter values accordingly.

        To update and trigger, use the event method.
        """
        self._set_stream_parameters(**kwargs)
        transformed = self.transform()
        if transformed:
            self._set_stream_parameters(**transformed)

    def __repr__(self):
        cls_name = self.__class__.__name__
        kwargs = ','.join('%s=%r' % (k,v)
                          for (k,v) in self.get_param_values() if k != 'name')
        if not self._rename:
            return '%s(%s)' % (cls_name, kwargs)
        else:
            return '%s(%r, %s)' % (cls_name, self._rename, kwargs)


    def __str__(self):
        return repr(self)


class Counter(Stream):
    """
    Simple stream that automatically increments an integer counter
    parameter every time it is updated.
    """

    counter = param.Integer(default=0, constant=True, bounds=(0,None))

    def transform(self):
        return {'counter': self.counter + 1}


class LinkedStream(Stream):
    """
    A LinkedStream indicates is automatically linked to plot interactions
    on a backend via a Renderer. Not all backends may support dynamically
    supplying stream data.
    """

    def __init__(self, linked=True, **params):
        super(LinkedStream, self).__init__(linked=linked, **params)


class PointerX(LinkedStream):
    """
    A pointer position along the x-axis in data coordinates which may be
    a numeric or categorical dimension.

    With the appropriate plotting backend, this corresponds to the
    position of the mouse/trackpad cursor. If the pointer is outside the
    plot bounds, the position is set to None.
    """

    x = param.ClassSelector(class_=(Number, util.basestring), default=None,
                            constant=True, doc="""
           Pointer position along the x-axis in data coordinates""")


class PointerY(LinkedStream):
    """
    A pointer position along the y-axis in data coordinates which may be
    a numeric or categorical dimension.

    With the appropriate plotting backend, this corresponds to the
    position of the mouse/trackpad pointer. If the pointer is outside
    the plot bounds, the position is set to None.
    """


    y = param.ClassSelector(class_=(Number, util.basestring), default=None,
                            constant=True, doc="""
           Pointer position along the y-axis in data coordinates""")


class PointerXY(LinkedStream):
    """
    A pointer position along the x- and y-axes in data coordinates which
    may numeric or categorical dimensions.

    With the appropriate plotting backend, this corresponds to the
    position of the mouse/trackpad pointer. If the pointer is outside
    the plot bounds, the position values are set to None.
    """

    x = param.ClassSelector(class_=(Number, util.basestring), default=None,
                            constant=True, doc="""
           Pointer position along the x-axis in data coordinates""")

    y = param.ClassSelector(class_=(Number, util.basestring), default=None,
                            constant=True, doc="""
           Pointer position along the y-axis in data coordinates""")


class Tap(PointerXY):
    """
    The x/y-position of a tap or click in data coordinates.
    """


class DoubleTap(PointerXY):
    """
    The x/y-position of a double-tap or -click in data coordinates.
    """


class MouseEnter(PointerXY):
    """
    The x/y-position where the mouse/cursor entered the plot area
    in data coordinates.
    """


class MouseLeave(PointerXY):
    """
    The x/y-position where the mouse/cursor entered the plot area
    in data coordinates.
    """


class PlotSize(LinkedStream):
    """
    Returns the dimensions of a plot once it has been displayed.
    """

    width = param.Integer(None, constant=True, doc="The width of the plot in pixels")

    height = param.Integer(None, constant=True, doc="The height of the plot in pixels")

    scale = param.Number(default=1.0, constant=True, doc="""
       Scale factor to scale width and height values reported by the stream""")

    def transform(self):
        return {'width':  int(self.width * self.scale),
                'height': int(self.height * self.scale)}


class RangeXY(LinkedStream):
    """
    Axis ranges along x- and y-axis in data coordinates.
    """

    x_range = param.NumericTuple(default=None, length=2, constant=True, doc="""
      Range of the x-axis of a plot in data coordinates""")

    y_range = param.NumericTuple(default=None, length=2, constant=True, doc="""
      Range of the y-axis of a plot in data coordinates""")


class RangeX(LinkedStream):
    """
    Axis range along x-axis in data coordinates.
    """

    x_range = param.NumericTuple(default=None, length=2, constant=True, doc="""
      Range of the x-axis of a plot in data coordinates""")


class RangeY(LinkedStream):
    """
    Axis range along y-axis in data coordinates.
    """

    y_range = param.NumericTuple(default=None, length=2, constant=True, doc="""
      Range of the y-axis of a plot in data coordinates""")


class Bounds(LinkedStream):
    """
    A stream representing the bounds of a box selection as an
    tuple of the left, bottom, right and top coordinates.
    """

    bounds = param.NumericTuple(default=None, constant=True, length=4,
                                allow_None=True, doc="""
        Bounds defined as (left, bottom, top, right) tuple.""")


class Selection1D(LinkedStream):
    """
    A stream representing a 1D selection of objects by their index.
    """

    index = param.List(default=[], constant=True, doc="""
        Indices into a 1D datastructure.""")


class ParamValues(Stream):
    """
    A Stream based on the parameter values of some other parameterized
    object, whether it is a parameterized class or a parameterized
    instance.

    The update method enables the stream to update the parameters of the
    specified object.
    """

    def __init__(self, obj, **params):
        self._obj = obj
        super(ParamValues, self).__init__(**params)


    @property
    def contents(self):
        if isinstance(self._obj, type):
            remapped={k: getattr(self._obj,k)
                               for k in self._obj.params().keys() if k!= 'name'}
        else:
            remapped={k:v for k,v in self._obj.get_param_values() if k!= 'name'}
        return remapped


    def update(self, **kwargs):
        """
        The update method updates the parameters of the specified object.

        If trigger is enabled, the trigger classmethod is invoked on
        this Stream instance to execute its subscribers.
        """
        if isinstance(self._obj, type):
            for name in self._obj.params().keys():
                if name in kwargs:
                    setattr(self._obj, name, kwargs[name])
        else:
            self._obj.set_param(**kwargs)

    def __repr__(self):
        cls_name = self.__class__.__name__
        return '%s(%r)' % (cls_name, self._obj)


    def __str__(self):
        return repr(self)


class PositionX(PointerX):
    def __init__(self, **params):
        self.warning('PositionX stream deprecated: use PointerX instead')
        super(PositionX, self).__init__(**params)

class PositionY(PointerY):
    def __init__(self, **params):
        self.warning('PositionY stream deprecated: use PointerY instead')
        super(PositionY, self).__init__(**params)

class PositionXY(PointerXY):
    def __init__(self, **params):
        self.warning('PositionXY stream deprecated: use PointerXY instead')
        super(PositionXY, self).__init__(**params)

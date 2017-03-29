"""
The streams module defines the streams API that allows visualizations to
generate and respond to events, originating either in Python on the
server-side or in Javascript in the Jupyter notebook (client-side).
"""

import param
from numbers import Number
from collections import defaultdict
from .core import util


class Stream(param.Parameterized):
    """
    A Stream is simply a parameterized object with parameters that
    change over time in response to update events. Parameters are
    updated via the update method.

    Streams may have one or more subscribers which are callables passed
    the parameter dictionary when the trigger classmethod is called.

    Depending on the plotting backend certain streams may interactively
    subscribe to events and changes by the plotting backend. To disable
    this behavior instantiate the Stream with linked=False.
    """

    # Mapping from a source id to a list of streams
    registry = defaultdict(list)

    # Mapping to define callbacks by backend and Stream type.
    # e.g. Stream._callbacks['bokeh'][Stream] = Callback
    _callbacks = defaultdict(dict)

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

        # Currently building a simple set of subscribers
        groups = [stream.subscribers for stream in streams]
        hidden = [stream._hidden_subscribers for stream in streams]
        subscribers = util.unique_iterator([s for subscribers in groups+hidden
                                            for s in subscribers])
        for subscriber in subscribers:
            subscriber(**dict(union))

        for stream in streams:
            stream.deactivate()


    def __init__(self, rename={}, source=None, linked=True, **params):
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
        self._hidden_subscribers = []
        self.linked = linked
        self._rename = self._validate_rename(rename)

        # The metadata may provide information about the currently
        # active event, i.e. the source of the stream values may
        # indicate where the event originated from
        self._metadata = {}

        super(Stream, self).__init__(**params)
        if source:
            self.registry[id(source)].append(self)


    @property
    def subscribers(self):
        return self._subscribers

    def clear(self):
        self._subscribers = []

    def add_subscriber(self, subscriber):
        if callable(subscriber):
            self._subscribers.append(subscriber)
        else:
            raise TypeError('Subscriber must be a callable.')

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


    def deactivate(self):
        """
        Allows defining an action after the stream has been triggered,
        e.g. resetting parameters on streams with transient events.
        """
        pass


    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        if self._source:
            raise Exception('source has already been defined on stream.')
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
        params = self.params().values()
        constants = [p.constant for p in params]
        for param in params:
            param.constant = False
        self.set_param(**kwargs)
        for (param, const) in zip(params, constants):
            param.constant = const

    def update(self, trigger=True, **kwargs):
        """
        The update method updates the stream parameters in response to
        some event. If the stream has a custom transform method, this
        is applied to transform the parameter values accordingly.

        If trigger is enabled, the trigger classmethod is invoked on
        this particular Stream instance.
        """
        self._set_stream_parameters(**kwargs)
        transformed = self.transform()
        if transformed:
            self._set_stream_parameters(**transformed)
        if trigger:
            self.trigger([self])


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


class PositionX(Stream):
    """
    A position along the x-axis in data coordinates.

    With the appropriate plotting backend, this may correspond to the
    position of the mouse/trackpad cursor.
    """

    x = param.ClassSelector(class_=(Number, util.basestring), default=0, doc="""
           Position along the x-axis in data coordinates""", constant=True)


class PositionY(Stream):
    """
    A position along the y-axis in data coordinates.

    With the appropriate plotting backend, this may correspond to the
    position of the mouse/trackpad cursor.
    """

    y = param.ClassSelector(class_=(Number, util.basestring), default=0, doc="""
           Position along the y-axis in data coordinates""", constant=True)


class PositionXY(Stream):
    """
    A position along the x- and y-axes in data coordinates.

    With the appropriate plotting backend, this may correspond to the
    position of the mouse/trackpad cursor.
    """

    x = param.ClassSelector(class_=(Number, util.basestring), default=0, doc="""
           Position along the x-axis in data coordinates""", constant=True)

    y = param.ClassSelector(class_=(Number, util.basestring), default=0, doc="""
           Position along the y-axis in data coordinates""", constant=True)


class Tap(PositionXY):
    """
    The x/y-position of a tap or click in data coordinates.
    """


class DoubleTap(PositionXY):
    """
    The x/y-position of a double-tap or -click in data coordinates.
    """


class MouseEnter(PositionXY):
    """
    The x/y-position where the mouse/cursor entered the plot area
    in data coordinates.
    """


class MouseLeave(PositionXY):
    """
    The x/y-position where the mouse/cursor entered the plot area
    in data coordinates.
    """


class PlotSize(Stream):
    """
    Returns the dimensions of a plot once it has been displayed.
    """

    width = param.Integer(300, constant=True, doc="The width of the plot in pixels")

    height = param.Integer(300, constant=True, doc="The height of the plot in pixels")

    scale = param.Number(default=1.0, constant=True, doc="""
       Scale factor to scale width and height values reported by the stream""")

    def transform(self):
        return {'width':  int(self.width * self.scale),
                'height': int(self.height * self.scale)}


class RangeXY(Stream):
    """
    Axis ranges along x- and y-axis in data coordinates.
    """

    x_range = param.NumericTuple(default=None, length=2, constant=True, doc="""
      Range of the x-axis of a plot in data coordinates""")

    y_range = param.NumericTuple(default=None, length=2, constant=True, doc="""
      Range of the y-axis of a plot in data coordinates""")


class RangeX(Stream):
    """
    Axis range along x-axis in data coordinates.
    """

    x_range = param.NumericTuple(default=None, length=2, constant=True, doc="""
      Range of the x-axis of a plot in data coordinates""")


class RangeY(Stream):
    """
    Axis range along y-axis in data coordinates.
    """

    y_range = param.NumericTuple(default=None, length=2, constant=True, doc="""
      Range of the y-axis of a plot in data coordinates""")


class Bounds(Stream):
    """
    A stream representing the bounds of a box selection as an
    tuple of the left, bottom, right and top coordinates.
    """

    bounds = param.NumericTuple(default=None, constant=True, length=4,
                                allow_None=True, doc="""
        Bounds defined as (left, bottom, top, right) tuple.""")


class Selection1D(Stream):
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


    def update(self, trigger=True, **kwargs):
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

        if trigger:
            self.trigger([self])


    def __repr__(self):
        cls_name = self.__class__.__name__
        return '%s(%r)' % (cls_name, self._obj)


    def __str__(self):
        return repr(self)

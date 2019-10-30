"""
The streams module defines the streams API that allows visualizations to
generate and respond to events, originating either in Python on the
server-side or in Javascript in the Jupyter notebook (client-side).
"""

import weakref
from numbers import Number
from collections import defaultdict
from contextlib import contextmanager
from itertools import groupby
from types import FunctionType

import param
import numpy as np

from .core import util
from .core.ndmapping import UniformNdMapping

# Types supported by Pointer derived streams
pointer_types = (Number, util.basestring, tuple)+util.datetime_types


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

    # Mapping from a source to a list of streams
    # WeakKeyDictionary to allow garbage collection
    # of unreferenced sources
    registry = weakref.WeakKeyDictionary()

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
        params = {'name': param.String(default=name)}
        for k, v in kwargs.items():
            kws = dict(default=v, constant=True)
            if isinstance(v, param.Parameter):
                params[k] = v
            elif isinstance(v, bool):
                params[k] = param.Boolean(**kws)
            elif isinstance(v, int):
                params[k] = param.Integer(**kws)
            elif isinstance(v, float):
                params[k] = param.Number(**kws)
            elif isinstance(v, str):
                params[k] = param.String(**kws)
            elif isinstance(v, dict):
                params[k] = param.Dict(**kws)
            elif isinstance(v, tuple):
                params[k] = param.Tuple(**kws)
            elif isinstance(v, list):
                params[k] = param.List(**kws)
            elif isinstance(v, np.ndarray):
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
        items = [stream.contents.items() for stream in set(streams)]
        union = [kv for kvs in items for kv in kvs]
        klist = [k for k, _ in union]
        key_clashes = set([k for k in klist if klist.count(k) > 1])
        if key_clashes:
            print('Parameter name clashes for keys %r' % key_clashes)

        # Group subscribers by precedence while keeping the ordering
        # within each group
        subscriber_precedence = defaultdict(list)
        for stream in streams:
            stream._on_trigger()
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


    def _on_trigger(self):
        """Called when a stream has been triggered"""


    @classmethod
    def _process_streams(cls, streams):
        """
        Processes a list of streams promoting Parameterized objects and
        methods to Param based streams.
        """
        parameterizeds = defaultdict(set)
        valid, invalid = [], []
        for s in streams:
            if isinstance(s, Stream):
                pass
            elif isinstance(s, param.Parameter):
                s = Params(s.owner, [s.name])
            elif isinstance(s, param.Parameterized):
                s = Params(s)
            elif util.is_param_method(s):
                if not hasattr(s, "_dinfo"):
                    continue
                s = ParamMethod(s)
            elif isinstance(s, FunctionType) and hasattr(s, "_dinfo"):
                deps = s._dinfo
                dep_params = list(deps['dependencies']) + list(deps.get('kw', {}).values())
                rename = {(p.owner, p.name): k for k, p in deps.get('kw', {}).items()}
                s = Params(parameters=dep_params, rename=rename)
            else:
                invalid.append(s)
                continue
            if isinstance(s, Params):
                pid = id(s.parameterized)
                overlap = (set(s.parameters) & parameterizeds[pid])
                if overlap:
                    pname = type(s.parameterized).__name__
                    param.main.param.warning(
                        'The %s parameter(s) on the %s object have '
                        'already been supplied in another stream. '
                        'Ensure that the supplied streams only specify '
                        'each parameter once, otherwise multiple '
                        'events will be triggered when the parameter '
                        'changes.' % (sorted([p.name for p in overlap]), pname))
                parameterizeds[pid] |= set(s.parameters)
            valid.append(s)
        return valid, invalid


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

        # Source is stored as a weakref to allow it to be garbage collected
        self._source = None if source is None else weakref.ref(source)

        self._subscribers = []
        for subscriber in subscribers:
            self.add_subscriber(subscriber)

        self.linked = linked
        self.transient = transient

        # Whether this stream is currently triggering its subscribers
        self._triggering = False

        # The metadata may provide information about the currently
        # active event, i.e. the source of the stream values may
        # indicate where the event originated from
        self._metadata = {}

        super(Stream, self).__init__(**params)
        self._rename = self._validate_rename(rename)
        if source is not None:
            if source in self.registry:
                self.registry[source].append(self)
            else:
                self.registry[source] = [self]


    @property
    def subscribers(self):
        """Property returning the subscriber list"""
        return [s for p, s in sorted(self._subscribers, key=lambda x: x[0])]


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
            remaining = [(p, s) for (p, s) in self._subscribers if p > 1]
        else:
            remaining = [(p, s) for (p, s) in self._subscribers if p <= 1]
        self._subscribers = remaining


    def reset(self):
        """
        Resets stream parameters to their defaults.
        """
        with util.disable_constant(self):
            for k, p in self.param.objects('existing').items():
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
        param_names = [k for k in self.param if k != 'name']
        for k, v in mapping.items():
            if k not in param_names:
                raise KeyError('Cannot rename %r as it is not a stream parameter' % k)
            if k != v and v in param_names:
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
        params = {k: v for k, v in self.get_param_values() if k != 'name'}
        return self.__class__(rename=mapping,
                              source=(self._source() if self._source else None),
                              linked=self.linked, **params)

    @property
    def source(self):
        return self._source() if self._source else None


    @source.setter
    def source(self, source):
        if self.source is not None:
            source_list = self.registry[self.source]
            if self in source_list:
                source_list.remove(self)
            if not source_list:
                self.registry.pop(self.source)

        if source is None:
            self._source = None
            return

        self._source = weakref.ref(source)
        if source in self.registry:
            self.registry[source].append(self)
        else:
            self.registry[source] = [self]


    def transform(self):
        """
        Method that can be overwritten by subclasses to process the
        parameter values before renaming is applied. Returns a
        dictionary of transformed parameters.
        """
        return {}


    @property
    def contents(self):
        filtered = {k: v for k, v in self.get_param_values() if k != 'name'}
        return {self._rename.get(k, k): v for (k, v) in filtered.items()
                if self._rename.get(k, True) is not None}

    @property
    def hashkey(self):
        """
        The object the memoization hash is computed from. By default
        returns the stream contents but can be overridden to provide
        a custom hash key.
        """
        return self.contents


    def _set_stream_parameters(self, **kwargs):
        """
        Sets the stream parameters which are expected to be declared
        constant.
        """
        with util.disable_constant(self):
            self.param.set_param(**kwargs)

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
        kwargs = ','.join('%s=%r' % (k, v)
                          for (k, v) in self.get_param_values() if k != 'name')
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

    counter = param.Integer(default=0, constant=True, bounds=(0, None))

    def transform(self):
        return {'counter': self.counter + 1}


class Pipe(Stream):
    """
    A Stream used to pipe arbitrary data to a callback.
    Unlike other streams memoization can be disabled for a
    Pipe stream (and is disabled by default).
    """

    data = param.Parameter(default=None, constant=True, doc="""
        Arbitrary data being streamed to a DynamicMap callback.""")

    def __init__(self, data=None, memoize=False, **params):
        super(Pipe, self).__init__(data=data, **params)
        self._memoize_counter = 0

    def send(self, data):
        """
        A convenience method to send an event with data without
        supplying a keyword.
        """
        self.event(data=data)

    def _on_trigger(self):
        self._memoize_counter += 1

    @property
    def hashkey(self):
        return {'_memoize_key': self._memoize_counter}


class Buffer(Pipe):
    """
    Buffer allows streaming and accumulating incoming chunks of rows
    from tabular datasets. The data may be in the form of a pandas
    DataFrame, 2D arrays of rows and columns or dictionaries of column
    arrays. Buffer will accumulate the last N rows, where N is defined
    by the specified ``length``. The accumulated data is then made
    available via the ``data`` parameter.

    A Buffer may also be instantiated with a streamz.StreamingDataFrame
    or a streamz.StreamingSeries, it will automatically subscribe to
    events emitted by a streamz object.

    When streaming a DataFrame will reset the DataFrame index by
    default making it available to HoloViews elements as dimensions,
    this may be disabled by setting index=False.

    The ``following`` argument determines whether any plot which is
    subscribed to this stream will update the axis ranges when an
    update is pushed. This makes it possible to control whether zooming
    is allowed while streaming.
    """

    def __init__(self, data, length=1000, index=True, following=True, **params):
        if (util.pd and isinstance(data, util.pd.DataFrame)):
            example = data
        elif isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError("Only 2D array data may be streamed by Buffer.")
            example = data
        elif isinstance(data, dict):
            if not all(isinstance(v, np.ndarray) for v in data.values()):
                raise ValueError("Data in dictionary must be of array types.")
            elif len(set(len(v) for v in data.values())) > 1:
                raise ValueError("Columns in dictionary must all be the same length.")
            example = data
        else:
            try:
                from streamz.dataframe import StreamingDataFrame, StreamingSeries
                loaded = True
            except ImportError:
                try:
                    from streamz.dataframe import DataFrame as StreamingDataFrame, Series as StreamingSeries
                    loaded = True
                except ImportError:
                    loaded = False
            if not loaded or not isinstance(data, (StreamingDataFrame, StreamingSeries)):
                raise ValueError("Buffer must be initialized with pandas DataFrame, "
                                 "streamz.StreamingDataFrame or streamz.StreamingSeries.")
            elif isinstance(data, StreamingSeries):
                data = data.to_frame()
            example = data.example
            data.stream.sink(self.send)
            self.sdf = data

        if index and (util.pd and isinstance(example, util.pd.DataFrame)):
            example = example.reset_index()
        params['data'] = example
        super(Buffer, self).__init__(**params)
        self.length = length
        self.following = following
        self._chunk_length = 0
        self._count = 0
        self._index = index


    def verify(self, x):
        """ Verify consistency of dataframes that pass through this stream """
        if type(x) != type(self.data):
            raise TypeError("Input expected to be of type %s, got %s." %
                            (type(self.data).__name__, type(x).__name__))
        elif isinstance(x, np.ndarray):
            if x.ndim != 2:
                raise ValueError('Streamed array data must be two-dimensional')
            elif x.shape[1] != self.data.shape[1]:
                raise ValueError("Streamed array data expeced to have %d columns, "
                                 "got %d." % (self.data.shape[1], x.shape[1]))
        elif util.pd and isinstance(x, util.pd.DataFrame) and list(x.columns) != list(self.data.columns):
            raise IndexError("Input expected to have columns %s, got %s" %
                             (list(self.data.columns), list(x.columns)))
        elif isinstance(x, dict):
            if any(c not in x for c in self.data):
                raise IndexError("Input expected to have columns %s, got %s" %
                                 (sorted(self.data.keys()), sorted(x.keys())))
            elif len(set(len(v) for v in x.values())) > 1:
                raise ValueError("Input columns expected to have the "
                                 "same number of rows.")


    def clear(self):
        "Clears the data in the stream"
        if isinstance(self.data, np.ndarray):
            data = self.data[:, :0]
        elif util.pd and isinstance(self.data, util.pd.DataFrame):
            data = self.data.iloc[:0]
        elif isinstance(self.data, dict):
            data = {k: v[:0] for k, v in self.data.items()}
        with util.disable_constant(self):
            self.data = data
        self.send(data)


    def _concat(self, data):
        """
        Concatenate and slice the accepted data types to the defined
        length.
        """
        if isinstance(data, np.ndarray):
            data_length = len(data)
            if data_length < self.length:
                prev_chunk = self.data[-(self.length-data_length):]
                data = np.concatenate([prev_chunk, data])
            elif data_length > self.length:
                data = data[-self.length:]
        elif util.pd and isinstance(data, util.pd.DataFrame):
            data_length = len(data)
            if data_length < self.length:
                prev_chunk = self.data.iloc[-(self.length-data_length):]
                data = util.pd.concat([prev_chunk, data])
            elif data_length > self.length:
                data = data.iloc[-self.length:]
        elif isinstance(data, dict) and data:
            data_length = len(list(data.values())[0])
            new_data = {}
            for k, v in data.items():
                if data_length < self.length:
                    prev_chunk = self.data[k][-(self.length-data_length):]
                    new_data[k] = np.concatenate([prev_chunk, v])
                elif data_length > self.length:
                    new_data[k] = v[-self.length:]
                else:
                    new_data[k] = v
            data = new_data
        self._chunk_length = data_length
        return data


    def update(self, **kwargs):
        """
        Overrides update to concatenate streamed data up to defined length.
        """
        data = kwargs.get('data')
        if data is not None:
            if (util.pd and isinstance(data, util.pd.DataFrame) and
                list(data.columns) != list(self.data.columns) and self._index):
                data = data.reset_index()
            self.verify(data)
            kwargs['data'] = self._concat(data)
            self._count += 1
        super(Buffer, self).update(**kwargs)


    @property
    def hashkey(self):
        return {'hash': self._count}



class Params(Stream):
    """
    A Stream that watches the changes in the parameters of the supplied
    Parameterized objects and triggers when they change.
    """

    parameterized = param.ClassSelector(class_=(param.Parameterized,
                                                param.parameterized.ParameterizedMetaclass),
                                        constant=True, allow_None=True, doc="""
        Parameterized instance to watch for parameter changes.""")

    parameters = param.List([], constant=True, doc="""
        Parameters on the parameterized to watch.""")

    def __init__(self, parameterized=None, parameters=None, watch=True, watch_only=False, **params):
        if util.param_version < '1.8.0' and watch:
            raise RuntimeError('Params stream requires param version >= 1.8.0, '
                               'to support watching parameters.')
        if parameters is None:
            parameters = [parameterized.param[p] for p in parameterized.param if p != 'name']
        else:
            parameters = [p if isinstance(p, param.Parameter) else parameterized.param[p]
                          for p in parameters]

        if 'rename' in params:
            rename = {}
            owners = [p.owner for p in parameters]
            for k, v in params['rename'].items():
                if isinstance(k, tuple):
                    rename[k] = v
                else:
                    rename.update({(o, k): v for o in owners})
            params['rename'] = rename

        self._watch_only = watch_only
        super(Params, self).__init__(parameterized=parameterized, parameters=parameters, **params)
        self._memoize_counter = 0
        self._events = []
        if watch:
            # Subscribe to parameters
            keyfn = lambda x: id(x.owner)
            for _, group in groupby(sorted(parameters, key=keyfn)):
                group = list(group)
                group[0].owner.param.watch(self._watcher, [p.name for p in group])

    @classmethod
    def from_params(cls, params):
        """Returns Params streams given a dictionary of parameters

        Args:
            params (dict): Dictionary of parameters

        Returns:
            List of Params streams
        """
        key_fn = lambda x: id(x[1].owner)
        streams = []
        for _, group in groupby(sorted(params.items(), key=key_fn), key_fn):
            group = list(group)
            inst = [p.owner for _, p in group][0]
            if not isinstance(inst, param.Parameterized):
                continue
            names = [p.name for _, p in group]
            rename = {p.name: n for n, p in group}
            streams.append(cls(inst, names, rename=rename))
        return streams

    def _validate_rename(self, mapping):
        pnames = [p.name for p in self.parameters]
        for k, v in mapping.items():
            n = k[1] if isinstance(k, tuple) else k
            if n not in pnames:
                raise KeyError('Cannot rename %r as it is not a stream parameter' % n)
            if n != v and v in pnames:
                raise KeyError('Cannot rename to %r as it clashes with a '
                               'stream parameter of the same name' % v)
        return mapping

    def _watcher(self, *events):
        try:
            self._events = list(events)
            self.trigger([self])
        except:
            raise
        finally:
            self._events = []

    def _on_trigger(self):
        if any(e.type == 'triggered' for e in self._events):
            self._memoize_counter += 1

    @property
    def hashkey(self):
        hashkey = {(p.owner, p.name): getattr(p.owner, p.name) for p in self.parameters}
        hashkey = {' '.join([o.name, self._rename.get((o, n), n)]): v for (o, n), v in hashkey.items()
                   if self._rename.get((o, n), True) is not None}
        hashkey['_memoize_key'] = self._memoize_counter
        return hashkey

    def reset(self):
        pass

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.parameterized, k, v)

    @property
    def contents(self):
        if self._watch_only:
            return {}
        filtered = {(p.owner, p.name): getattr(p.owner, p.name) for p in self.parameters}
        return {self._rename.get((o, n), n): v for (o, n), v in filtered.items()
                if self._rename.get((o, n), True) is not None}



class ParamMethod(Params):
    """
    A Stream that watches the parameter dependencies on a method of
    a parameterized class and triggers when one of the parameters
    change.
    """

    def __init__(self, parameterized, parameters=None, watch=True, **params):
        if not util.is_param_method(parameterized):
            raise ValueError('ParamMethod stream expects a method on a '
                             'parameterized class, found %s.'
                             % type(parameterized).__name__)
        method = parameterized
        parameterized = util.get_method_owner(parameterized)
        if not parameters:
            parameters = [p.pobj for p in parameterized.param.params_depended_on(method.__name__)]
        params['watch_only'] = True
        super(ParamMethod, self).__init__(parameterized, parameters, watch, **params)




# Backward compatibility
def ParamValues(*args, **kwargs):
    param.main.param.warning('ParamValues stream is deprecated, use Params stream instead.')
    kwargs['watch'] = False
    return Params(*args, **kwargs)


class SelectionExpr(Stream):

    selection_expr = param.Parameter(default=None, constant=True)
    bbox = param.Dict(default=None, constant=True)

    def __init__(self, source, **params):
        from .element import Element
        from .core.spaces import DynamicMap
        from .plotting.util import initialize_dynamic

        if isinstance(source, DynamicMap):
            initialize_dynamic(source)

        if isinstance(source, Element) or (
                isinstance(source, DynamicMap) and
                issubclass(source.type, Element)
        ):
            self._source_streams = []
            super(SelectionExpr, self).__init__(source=source, **params)
            self._register_chart(source)
        else:
            raise ValueError("""
The source of SelectionExpr must be an instance of an Element subclass,
or a DynamicMap that returns such an instance
            Received value of type {typ}: {val}""".format(
            typ=type(source), val=source
        ))

    def _register_chart(self, hvobj):
        from .core.spaces import DynamicMap

        if isinstance(hvobj, DynamicMap):
            element_type = hvobj.type
        else:
            element_type = hvobj

        selection_streams = element_type._selection_streams

        def _set_expr(**params):
            if isinstance(hvobj, DynamicMap):
                element = hvobj.values()[-1]
            else:
                element = hvobj
            selection_expr, bbox = \
                element._get_selection_expr_for_stream_value(**params);

            self.event(selection_expr=selection_expr, bbox=bbox)

        for stream_type in selection_streams:
            stream = stream_type(source=hvobj)
            self._source_streams.append(stream)

            stream.add_subscriber(_set_expr)

    def _unregister_chart(self):
        for stream in self._source_streams:
            stream.source = None
            stream.clear()
        self._source_streams.clear()

    @property
    def source(self):
        return Stream.source.fget(self)

    @source.setter
    def source(self, value):
        self._unregister_chart()
        Stream.source.fset(self, value)

    def __del__(self):
        self._unregister_chart()

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

    x = param.ClassSelector(class_=pointer_types, default=None,
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


    y = param.ClassSelector(class_=pointer_types, default=None,
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

    x = param.ClassSelector(class_=pointer_types, default=None,
                            constant=True, doc="""
           Pointer position along the x-axis in data coordinates""")

    y = param.ClassSelector(class_=pointer_types, default=None,
                            constant=True, doc="""
           Pointer position along the y-axis in data coordinates""")


class Draw(PointerXY):
    """
    A series of updating x/y-positions when drawing, together with the
    current stroke count
    """

    stroke_count = param.Integer(default=0, constant=True, doc="""
       The current drawing stroke count. Increments every time a new
       stroke is started.""")

class SingleTap(PointerXY):
    """
    The x/y-position of a single tap or click in data coordinates.
    """


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

    x_range = param.Tuple(default=None, length=2, constant=True, doc="""
      Range of the x-axis of a plot in data coordinates""")

    y_range = param.Tuple(default=None, length=2, constant=True, doc="""
      Range of the y-axis of a plot in data coordinates""")


class RangeX(LinkedStream):
    """
    Axis range along x-axis in data coordinates.
    """

    x_range = param.Tuple(default=None, length=2, constant=True, doc="""
      Range of the x-axis of a plot in data coordinates""")


class RangeY(LinkedStream):
    """
    Axis range along y-axis in data coordinates.
    """

    y_range = param.Tuple(default=None, length=2, constant=True, doc="""
      Range of the y-axis of a plot in data coordinates""")


class BoundsXY(LinkedStream):
    """
    A stream representing the bounds of a box selection as an
    tuple of the left, bottom, right and top coordinates.
    """

    bounds = param.Tuple(default=None, constant=True, length=4,
                         allow_None=True, doc="""
        Bounds defined as (left, bottom, right, top) tuple.""")


class BoundsX(LinkedStream):
    """
    A stream representing the bounds of a box selection as an
    tuple of the left and right coordinates.
    """

    boundsx = param.Tuple(default=None, constant=True, length=2,
                          allow_None=True, doc="""
        Bounds defined as (left, right) tuple.""")


class BoundsY(LinkedStream):
    """
    A stream representing the bounds of a box selection as an
    tuple of the bottom and top coordinates.
    """

    boundsy = param.Tuple(default=None, constant=True, length=2,
                          allow_None=True, doc="""
        Bounds defined as (bottom, top) tuple.""")


class Selection1D(LinkedStream):
    """
    A stream representing a 1D selection of objects by their index.
    """

    index = param.List(default=[], constant=True, doc="""
        Indices into a 1D datastructure.""")


class PlotReset(LinkedStream):
    """
    A stream signalling when a plot reset event has been triggered.
    """

    resetting = param.Boolean(default=False, constant=True, doc="""
        Whether a reset event is being signalled.""")

    def __init__(self, *args, **params):
        super(PlotReset, self).__init__(self, *args, **dict(params, transient=True))


class CDSStream(LinkedStream):
    """
    A Stream that syncs a bokeh ColumnDataSource with python.
    """

    data = param.Dict(constant=True, doc="""
        Data synced from Bokeh ColumnDataSource supplied as a
        dictionary of columns, where each column is a list of values
        (for point-like data) or list of lists of values (for
        path-like data).""")


class PointDraw(CDSStream):
    """
    Attaches a PointAddTool and syncs the datasource.

    drag: boolean
        Whether to enable dragging of polygons and paths

    empty_value: int/float/string/None
        The value to insert on non-position columns when adding a new polygon

    num_objects: int
        The number of polygons that can be drawn before overwriting
        the oldest polygon.

    styles: dict
    A dictionary specifying lists of styles to cycle over whenever
    a new Point glyph is drawn.
    """

    def __init__(self, empty_value=None, drag=True, num_objects=0, styles={}, **params):
        self.drag = drag
        self.empty_value = empty_value
        self.num_objects = num_objects
        self.styles = styles
        super(PointDraw, self).__init__(**params)

    @property
    def element(self):
        source = self.source
        if isinstance(source, UniformNdMapping):
            source = source.last
        if not self.data:
            return source.clone([], id=None)
        return source.clone(self.data, id=None)

    @property
    def dynamic(self):
        from .core.spaces import DynamicMap
        return DynamicMap(lambda *args, **kwargs: self.element, streams=[self])



class PolyDraw(CDSStream):
    """
    Attaches a PolyDrawTool and syncs the datasource.

    drag: boolean
        Whether to enable dragging of polygons and paths

    empty_value: int/float/string/None
        The value to insert on non-position columns when adding a new polygon

    num_objects: int
        The number of polygons that can be drawn before overwriting
        the oldest polygon.

    show_vertices: boolean
        Whether to show the vertices when a polygon is selected

    vertex_style: dict
        A dictionary specifying the style options for the vertices.
        The usual bokeh style options apply, e.g. fill_color,
        line_alpha, size, etc.

    styles: dict
        A dictionary specifying lists of styles to cycle over whenever
        a new Poly glyph is drawn.
    """

    def __init__(self, empty_value=None, drag=True, num_objects=0,
                 show_vertices=False, vertex_style={}, styles={},
                 **params):
        self.drag = drag
        self.empty_value = empty_value
        self.num_objects = num_objects
        self.show_vertices = show_vertices
        self.vertex_style = vertex_style
        self.styles = styles
        super(PolyDraw, self).__init__(**params)

    @property
    def element(self):
        source = self.source
        if isinstance(source, UniformNdMapping):
            source = source.last
        data = self.data
        if not data:
            return source.clone([], id=None)
        cols = list(self.data)
        x, y = source.kdims
        lookup = {'xs': x.name, 'ys': y.name}
        data = [{lookup.get(c, c): data[c][i] for c in self.data}
                for i in range(len(data[cols[0]]))]
        return source.clone(data, id=None)

    @property
    def dynamic(self):
        from .core.spaces import DynamicMap
        return DynamicMap(lambda *args, **kwargs: self.element, streams=[self])


class FreehandDraw(CDSStream):
    """
    Attaches a FreehandDrawTool and syncs the datasource.

    empty_value: int/float/string/None
        The value to insert on non-position columns when adding a new polygon

    num_objects: int
        The number of polygons that can be drawn before overwriting
        the oldest polygon.

    styles: dict
        A dictionary specifying lists of styles to cycle over whenever
        a new freehand glyph is drawn.
    """

    def __init__(self, empty_value=None, num_objects=0, styles={}, **params):
        self.empty_value = empty_value
        self.num_objects = num_objects
        self.styles = styles
        super(FreehandDraw, self).__init__(**params)

    @property
    def element(self):
        source = self.source
        if isinstance(source, UniformNdMapping):
            source = source.last
        data = self.data
        if not data:
            return source.clone([], id=None)
        cols = list(self.data)
        x, y = source.kdims
        lookup = {'xs': x.name, 'ys': y.name}
        data = [{lookup.get(c, c): data[c][i] for c in self.data}
                for i in range(len(data[cols[0]]))]
        return source.clone(data, id=None)

    @property
    def dynamic(self):
        from .core.spaces import DynamicMap
        return DynamicMap(lambda *args, **kwargs: self.element, streams=[self])



class BoxEdit(CDSStream):
    """
    Attaches a BoxEditTool and syncs the datasource.

    empty_value: int/float/string/None
        The value to insert on non-position columns when adding a new box

    num_objects: int
        The number of boxes that can be drawn before overwriting the
        oldest drawn box.

    styles: dict
        A dictionary specifying lists of styles to cycle over whenever
        a new box glyph is drawn.
    """

    def __init__(self, empty_value=None, num_objects=0, styles={}, **params):
        self.empty_value = empty_value
        self.num_objects = num_objects
        self.styles = styles
        super(BoxEdit, self).__init__(**params)

    @property
    def element(self):
        from .element import Polygons
        source = self.source
        if isinstance(source, UniformNdMapping):
            source = source.last
        data = self.data
        if not data:
            return source.clone([])
        paths = []
        for (x0, x1, y0, y1) in zip(data['x0'], data['x1'], data['y0'], data['y1']):
            xs = [x0, x0, x1, x1]
            ys = [y0, y1, y1, y0]
            if isinstance(source, Polygons):
                xs.append(x0)
                ys.append(y0)
            paths.append(np.column_stack((xs, ys)))
        return source.clone(paths)

    @property
    def dynamic(self):
        from .core.spaces import DynamicMap
        return DynamicMap(lambda *args, **kwargs: self.element, streams=[self])



class PolyEdit(PolyDraw):
    """
    Attaches a PolyEditTool and syncs the datasource.

    shared: boolean
        Whether PolyEditTools should be shared between multiple elements

    vertex_style: dict
        A dictionary specifying the style options for the vertices.
        The usual bokeh style options apply, e.g. fill_color,
        line_alpha, size, etc.
    """

    def __init__(self, vertex_style={}, shared=True, **params):
        self.shared = shared
        super(PolyEdit, self).__init__(vertex_style=vertex_style, **params)

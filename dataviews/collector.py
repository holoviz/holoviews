"""
AttributeTree, Collector and related classes offer optional functionality
for holding and collecting DataView objects.
"""
import uuid
import numpy as np
from collections import OrderedDict

import param
from .sheetviews import CoordinateGrid
from .sheetviews import SheetView  # pyflakes:ignore (Needed for doctests)
from .views import GridLayout, Stack, View, NdMapping, Dimension

from .ipython.widgets import RunProgress

Time = Dimension("Time", type=param.Dynamic.time_fn.time_type)


class AttrTree(object):
    """
    An AttrTree offers convenient, multi-level attribute access for
    collections of objects. AttrTree objects may also be combined
    together using the update method or merge classmethod. Here is an
    example of adding a View to an AttrTree and accessing it:

    >>> t = AttrTree()
    >>> t.Example.Path = View('data1')
    >>> t.Example.Path                             #doctest: +ELLIPSIS
    View('data1', ...)
    """

    @classmethod
    def merge(cls, trees):
        """
        Merge a collection of AttrTree objects.
        """
        first = trees[0]
        for tree in trees:
            first.update(tree)
        return first

    def __init__(self, label=None, parent=None):
        self.__dict__['parent'] = parent
        self.__dict__['label'] = label
        self.__dict__['children'] = []
        # Path items will only be populated at root node
        self.__dict__['path_items'] = OrderedDict()

        self.__dict__['_fixed'] = False
        fixed_error = 'AttrTree attribute access disabled with fixed=True'
        self.__dict__['_fixed_error'] = fixed_error

    @property
    def fixed(self):
        "If fixed, no new paths can be created via attribute access"
        return self.__dict__['_fixed']

    @fixed.setter
    def fixed(self, val):
        self.__dict__['_fixed'] = val


    def grid(self, ordering='alphanumeric'):
        """
        Turn the AttrTree into a GridLayout with the available View
        objects ordering specified by a list of labels or by the
        specified ordering mode ('alphanumeric' or 'insertion').
        """
        if ordering == 'alphanumeric':
            child_ordering = sorted(self.children)
        elif ordering == 'insertion':
            child_ordering = self.children
        else:
            child_ordering = ordering

        children = [self.__dict__[l] for l in child_ordering]
        dataview_types = (View, Stack, GridLayout, CoordinateGrid)
        return GridLayout(list(child for child in children
                               if isinstance(child, dataview_types)))


    def update(self, other):
        """
        Updated the contents of the current AttrTree with the
        contents of a second AttrTree.
        """
        fixed_status = (self.fixed, other.fixed)
        (self.fixed, other.fixed) = (False, False)
        if self.parent is None:
            self.path_items.update(other.path_items)
        for label in other.children:
            item = other[label]
            if label not in self:
                self[label] = item
            else:
                self[label].update(item)
        (self.fixed, other.fixed) =  fixed_status


    def set_path(self, path, val):
        """
        Set the given value at the supplied path where path is either
        a tuple of strings or a string in A.B.C format.
        """
        path = tuple(path.split('.')) if isinstance(path , str) else tuple(path)
        if not all(p[0].isupper() for p in path):
            raise Exception("All paths elements must be capitalized.")

        if len(path) > 1:
            attrtree = self.__getattr__(path[0])
            attrtree.set_path(path[1:], val)
        else:
            self.__setattr__(path[0], val)


    def _propagate(self, path, val):
        """
        Propagate the value to the root node.
        """
        if self.parent is None: # Root
            self.path_items[path] = val
        else:
            self.parent._propagate((self.label,)+path, val)


    def __setitem__(self, label, val):
        self.__setattr__(label, val)


    def __getitem__(self, label):
        """
        Access a child element by label.
        """
        return self.__dict__[label]


    def __setattr__(self, label, val):
        # Getattr is skipped for root and first set of children
        shallow = (self.parent is None or self.parent.parent is None)
        if label[0].isupper() and self.fixed and shallow:
            raise AttributeError(self._fixed_error)

        super(AttrTree, self).__setattr__(label, val)

        if label in self.children: pass
        elif label[0].isupper():
            self.children.append(label)
            self._propagate((label,), val)


    def __getattr__(self, label):
        """
        Access a label from the AttrTree or generate a new AttrTree
        with the chosen attribute path.
        """
        try:
            return super(AttrTree, self).__getattr__(label)
        except AttributeError: pass

        if label.startswith('_'):   raise AttributeError(str(label))
        elif self.fixed==True:      raise AttributeError(self._fixed_error)

        if label in self.children:
            return self.__dict__[label]

        if label[0].isupper():
            self.children.append(label)
            child_tree = AttrTree(label=label, parent=self)
            self.__dict__[label] = child_tree
            return child_tree
        else:
            raise AttributeError("%s: Custom paths elements must be capitalized." % label)


    def __repr__(self):
        """
        A useful summary of the contents of the AttrTree node that
        works for any node in the tree. Note that this is not a repr
        that can be evaluated.
        """
        path, node = [], self
        while node.parent is not None:
            path = [node.label] + path
            node = node.parent

        filtered = OrderedDict([(k,v) for (k,v) in node.path_items.items()
                                if k[:len(path)]==tuple(path)])
        if len(filtered) == 0:
            return "Dangling AttrTree node with no leaf items."

        path_strs = ['.'.join(p) for p in filtered]
        max_chars = max(len(el) for el in path_strs)

        lines = ["AttrTree with %d leaf nodes of type:\n" % len(filtered)]
        for (path_str, val) in zip(path_strs, filtered.values()):
            val_type = type(val).__name__
            lines.append("   %s : %s" % (path_str.ljust(max_chars), val_type))
        return "\n".join(lines)


    def __contains__(self, name):
        return name in self.children or name in self.path_items



class Reference(object):
    """
    A Reference allows access to an object to be deferred until it is
    needed in the appropriate context. References are used by
    Collector to capture the state of an object at collection time.

    One particularly important property of references is that they
    should be pickleable. This means that you can pickle Collectors so
    that you can unpickle them in different environments and still
    collect from the required object.

    A Reference only needs to have a resolved_type property and a
    resolve method. The constructor will take some specification of
    where to find the target object (may be the object itself).
    """

    @property
    def resolved_type(self):
        """
        Returns the type of the object resolved by this references. If
        multiple types are possible, the return is a tuple of types.
        """
        raise NotImplementedError


    def resolve(self, container=None):
        """
        Return the referenced object. Optionally, a container may be
        passed in from which the object is to be resolved.
        """
        raise NotImplementedError



class ViewRef(Reference):
    """
    A ViewRef object is a Reference to a dataview object in an
    Attrtree that may not exist when initialized. This makes it
    possible to schedule tasks for processing data not yet present.

    ViewRefs compose with the * operator to specify Overlays and also
    support slicing of the referenced view objects:

    >>> ref = ViewRef('Example.Path1 * Example.Path2')

    >>> tree = AttrTree()
    >>> tree.Example.Path1 = SheetView(np.random.rand(5,5))
    >>> tree.Example.Path2 = SheetView(np.random.rand(5,5))
    >>> overlay = ref.resolve(tree)
    >>> len(overlay)
    2

    Note that the operands of * must be distinct ViewRef objects.
    """
    def __init__(self, spec=''):
        """
        The specification is a string that follows attribute access on
        an AttrTree. The '*' operator is supported, as well as slicing
        syntax - an example specification is 'A.B.C[2:4] *D.E'
        """
        self.specification, self.slices = self._parse_spec(spec)
        if not all(p[0].isupper() for path in self.specification for p in path):
            raise Exception("All path components must be capitalized.")

    @property
    def spec(self):
        paths = ['.'.join(s for s in spec) for spec in self.specification]
        indices = [self.slices.get(spec, None) for spec in self.specification]
        indexed_paths = [p + self._pprint_index(s) for (p,s) in zip(paths, indices)]
        return ' * '.join(indexed_paths)


    def _parse_spec(self, spec):

        class Index(object):
            def __getitem__(self, val):
                return val

        specs, slices = [], {}
        if spec.strip() == '':
            return specs, slices
        components = [el.strip() for el in spec.split('*')]
        for component in components:
            if component.count('[') !=  component.count(']'):
                raise Exception("Mismatched parentheses in %r" % component)
            elif component.count('[') in [0,1]:
                path_spec = tuple(component.split('.'))
                specs.append(path_spec)
            else:
                raise Exception("Invalid syntax %r" % component)

            if component.count('[') == 1:
                opening = component.find('[')
                closing = component.find(']')
                path_spec = tuple(component[:opening].split('.'))
                if opening > closing:
                    raise Exception("Invalid syntax %r" % component)
                slices[path_spec] = eval('Index()[%s]' % component[opening+1:closing])
            else:
                slices[path_spec] = None
        return specs, slices


    @property
    def resolved_type(self):
        return (View, Stack, CoordinateGrid)


    def _resolve_ref(self, ref, attrtree):
        """
        Get the View referred to by a single reference tuple if the
        data exists, otherwise raise AttributeError.
        """
        obj = attrtree
        for label in ref:
            if label in obj:
                obj= obj[label]
            else:
                info = ('.'.join(ref), label)
                raise AttributeError("Could not resolve %r at level %r" % info)
        return obj


    def resolve(self, attrtree):
        """
        Resolve the current ViewRef object into the appropriate View
        object (if available).
        """
        overlaid_view = None
        for ref in self.specification:
            view = self._resolve_ref(ref, attrtree)
            # Access specified slices for the view
            slc = self.slices.get(ref, None)
            view = view if slc is None else view[slc]

            if overlaid_view is None:
                overlaid_view = view
            else:
                overlaid_view = overlaid_view * view
        return overlaid_view


    def __getitem__(self, index):
        """
        Slice the referenced DataView.
        """
        for ref in self.specification:
            self.slices[ref] = index
        return self


    def __getattr__(self, label):
        """
        Multi-level attribute access on a ViewRef() object creates a
        reference with the same specified attribute access path.
        """
        try:
            return super(ViewRef, self).__getattr__(label)
        except AttributeError:

            if not label[0].isupper():
                raise AttributeError("Reference path element %r must capitalized" % label)
            elif len(self.specification) > 1:
                raise AttributeError("Cannot use attribute specification for overlays.")

        if len(self.specification) == 0:
            self.specification = [(label,)]
        elif len(self.specification) == 1:
            self.specification = [self.specification[0] + (label,)]
        return self


    def __mul__(self, other):
        """
        ViewRef object can be composed in to overlays.
        """
        if id(self) == id(other):
            raise Exception("Please ensure that each operand are distinct ViewRef objects.")
        return ViewRef(self.spec + ' * ' + other.spec)


    def _pprint_index(self, ind):
        if isinstance(ind, slice):
            parts = [str(el) for el in [ind.start, ind.stop, ind.step]]
            parts = parts[:2] if parts[2]=='None' else parts
            return '[%s]' % ':'.join(el if el!='None' else '' for el in parts)
        elif ind is None:
            return ''
        else:
            return '[%r]' % ind

    def __repr__(self):
        return 'ViewRef(%r)' % self.spec

    def __len__(self):
        return len(self.specification)




class Collect(object):
    """
    An Collect takes an object and corresponding hook and when
    called with an AttrTree, updates it with the output of the hook
    (given the object). The output of the hook should be a View or an
    AttrTree.

    The input object may be a picklable object (e.g. a
    ParameterizedFunction) or a Reference to the target object.  The
    supplied *args and **kwargs are passed to the hook together with
    the resolved object.

    When mode is 'merge' the return value of the hook needs to be an
    AttrTree to be merged with the attrtree when called.
    """

    @classmethod
    def select_hook(cls, obj, hooks):
        """
        Select the most appropriate hook by the most specific type.
        """
        matches = []
        obj_class = obj if isinstance(obj, type) else type(obj)

        if obj_class == param.parameterized.ParameterizedMetaclass:
            obj_class = obj

        for tp in hooks.keys():
            if issubclass(obj_class, tp):
                matches.append(tp)

        if len(matches) == 0:
            raise Exception("No hook found for object of type %s"
                            % obj.__class__.__name__)

        for obj_cls in obj_class.mro():
            if obj_cls in matches:
                return hooks[obj_cls]

        raise Exception("Match not in object classes mro()")


    def __init__(self, obj, *args, **kwargs):

        self.args=list(args)
        if 'times' in kwargs:
            self.times = kwargs.pop('times')
        else:
            self.times = []
        self.kwargs=kwargs
        self.path = None
        resolveable = None
        if hasattr(obj, 'resolve'):
            resolveable = obj
            obj = obj.resolved_type

        self.hook, self.mode, resolver = self.select_hook(obj, Collector.type_hooks)
        if resolveable is None:
            resolveable = obj if resolver is None else resolver(obj)
        self.obj = resolveable



    def _get_result(self, attrtree, time, times):
        """
        Method returning a View or AttrTree to be merged into the
        attrtree (via the specified hook) in the call.
        """
        resolvable = hasattr(self.obj, 'resolve')
        obj = self.obj.resolve() if resolvable else self.obj
        return self.hook(obj, *self.args, **self.kwargs)


    def __call__(self, attrtree, time=None, times=None):
        """
        Update and return the supplied AttrTree with the output of
        the hook at the given time out of the given list of times.
        """
        if self.path is None:
            raise Exception("Aggregation path not set.")

        if self.times and time not in self.times:
            return attrtree

        val = self._get_result(attrtree, time, times)
        if val is None:  return attrtree

        if self.mode == 'merge':
            if isinstance(val, AttrTree):
                attrtree.update(val)
                return attrtree
            else:
                raise Exception("Return value is not a AttrTree and mode is 'merge'.")

        if self.path not in attrtree:
            if not isinstance(val, NdMapping):
                val = val.stack_type([((time,), val)], dimensions=[Time])
        else:
            current_val = attrtree.path_items[self.path]
            val = self._merge_views(current_val, val, time)

        attrtree.set_path(self.path,  val)
        return attrtree


    def _merge_views(self, current_val, val, time):
        """
        Helper for merging views together. For instance, this method
        will add a SheetView to a SheetStack or merge two SheetStacks.
        """
        if isinstance(val, View):
            current_val[time] = val
        elif (isinstance(current_val, Stack) and 'Time' not in current_val.dimension_labels):
            raise Exception("Time dimension is missing.")
        else:
            current_val.update(val)
        return current_val

    def __repr__(self):
        args = ', '.join(str(el) for el in self.args) if self.args else ''
        kwargs = ', '.join('%s=%r' % (k,v) for (k,v) in self.kwargs.items()) if self.kwargs else ''
        name = self.obj.name if hasattr(self.obj, 'name') else repr(self.obj)
        return 'Collect(%s%s%s)' %  (name,
                                        (', %s' % args) if args else '',
                                        (', %s' % kwargs) if kwargs else '')

    def __str__(self):
        if hasattr(self.obj, 'name'):
            obj_name = self.obj.name
        else:
            obj_name = self.obj

        args = [str(el) for el in self.args]+['%s=%r' % (k,v) for (k,v) in self.kwargs.items()]
        arguments = '' if len(args)==0 else ' [%s]' % ','.join(args)
        return "%s%s" % (obj_name, arguments)


class Analyze(Collect):
    """
    An Analyze is a type of Collect that updates an Attrtree with
    the results of a ViewOperation. Analyze takes a ViewRef object as
    input which is resolved to generate input for the ViewOperation.
    """

    def __init__(self, reference, analysis, *args, **kwargs):
        self.reference = reference
        self.analysis = analysis

        self.args = list(args)
        if 'times' in kwargs:
            self.times = kwargs.pop('times')
        else:
            self.times = []
        self.kwargs = kwargs
        self.stackwise = self.kwargs.pop('stackwise', False)
        self.mode = 'set'
        self.path = None


    def _get_result(self, attrtree, time, times):
        if self.stackwise and time==times[-1]:
            view = self.reference.resolve(attrtree)
            return self.analysis(view, *self.args, **self.kwargs)
        elif self.stackwise:
            return None
        else:
            view = self.reference.resolve(attrtree)
            return self.analysis(view, *self.args, **self.kwargs)

    def __repr__(self):
        args = ', '.join(str(el) for el in self.args) if self.args else ''
        kwargs = ', '.join('%s=%r' % (k,v) for (k,v) in self.kwargs.items()) if self.kwargs else ''
        return "Analyze(%r, %s%s%s)" % (self.reference, self.analysis.name,
                                         (', %s' % args) if args else '',
                                         (', %s' % kwargs) if kwargs else '')


    def __str__(self):
        return "%s(%s)" % (self.analysis.name, self.reference)





class Collector(AttrTree):
    """
    A Collector specifies a template for how to populate a AttrTree
    with data over time. Two methods are used to schedule data
    collection: 'collect' and 'analyze'.

    The collect method takes an object (or reference) and collects
    views from it (as configured by setting an appropriate hook set
    with the for_type classmethod).

    The analysis method takes a reference to data on the attrtree (a
    ViewRef) and passes the resolved output to the given analysisfn
    ViewOperation.

    >>> Collector.for_type(str, lambda x: View(x, name=x))
    >>> Collector.interval_hook = param.Dynamic.time_fn.advance

    >>> c = Collector()
    >>> c.Target.Path = c.collect('example string')

    # Start collection...
    >>> data = c(times=[1,2,3,4,5])
    >>> isinstance(data, AttrTree)
    True
    >>> isinstance(data.Target.Path, Stack)
    True

    >>> times = data.Target.Path.keys()
    >>> print("Collected the data for %d time values" % len(times))
    Collected the data for 5 time values

    >>> data.Target.Path.last                 #doctest: +ELLIPSIS
    View('example string'...)
    """

    # A callable that advances by the specified time before the next
    # batch of collection tasks is executed. If set to a subclass of
    # RunProgress, the class will be instantiated and precent_range
    # updated to allow a progress bar to be displayed
    interval_hook = RunProgress


    # A callable that returns the time where the time may be the
    # simulation time or wall-clock time. The time values are
    # recorded by the Stack keys
    time_fn = param.Dynamic.time_fn

    type_hooks = {}

    @classmethod
    def for_type(cls, tp, hookfn, referencer=None, mode='set'):
        """
        For an object of a given type, apply the hookfn and use the
        specified mode to aggregate the data.

        To allow pickling (or any other defered access) of the target
        object, a referencer (a Reference subclass) may be specified
        to wrap the object as required.

        If mode is 'merge', merge the AttrTree output by the hook,
        otherwise if 'set', add the output to the path specified by
        the ViewRef.
        """
        cls.type_hooks[tp] = (hookfn, mode, referencer)


    def __init__(self, specs=[], **kwargs):
        super(Collector,self).__init__(**kwargs)

        for (path_spec, obj) in specs:
            if path_spec is None:
                self.__dict__['path_items'][uuid.uuid4().hex] = obj
            else:
                path = path_spec.rsplit('.')
                self.set_path(path, obj)

        self._scheduled_tasks = []

        fixed_error = 'Collector specification disabled after first call.'
        self.__dict__['_fixed_error'] = fixed_error
        self.__dict__['progress_label'] = 'Completion'


    @property
    def ref(self):
        """
        A convenient property to easily generate ViewRef object (via
        attribute access). Used to define View references for analysis
        or for setting a path for an Collect on the Collector.
        """
        return ViewRef()


    def collect(self, obj, *args, **kwargs):
        """
        Aggregate views from the object at each step by passing the
        arguments to the corresponding hook. The object may represent
        itself, or it may be a Reference. If a referencer class was
        specified when the hook was defined, the object will
        automatically be wrapped into a reference.
        """
        task = Collect(obj, *args, **kwargs)
        if task.mode == 'merge':
            self.path_items[uuid.uuid4().hex] = task
        return task



    def analyze(self, reference, analysisfn,  *args, **kwargs):
        """
        Given a ViewRef and the ViewOperation analysisfn, process the
        data resolved by the reference with analysisfn at each step.
        """
        return Analyze(reference, analysisfn, *args, **kwargs)


    def __call__(self, attrtree=AttrTree(), times=[]):

        current_time = self.time_fn()
        if times != sorted(times):
            raise Exception("Please supply the list of times in ascending order")
        if times[0] < current_time:
            raise Exception("The first time value is prior to the current time.")
        self.verify_times(times)

        times = np.array([current_time] + times)
        if len(set(times)) == 1:
            completion = [0,100]
        else:
            completion = 100 * (times - times.min()) / (times.max() - times.min())

        update_progress = (isinstance(self.interval_hook, type)
                           and issubclass(self.interval_hook, RunProgress))

        # If an instance of RunProgress, instantiate the progress bar
        interval_hook = (self.interval_hook(label=self.progress_label)
                         if update_progress else self.interval_hook)

        self._schedule_tasks()
        (self.fixed, attrtree.fixed) = (False, False)

        for i, t in enumerate(np.diff(times)):
            if update_progress:
                interval_hook.percent_range = (completion[i], completion[i+1])

            interval_hook(float(t))

            # An empty attrtree buffer stops analysis repeatedly
            # computing results over the entire accumulated stack
            attrtree_buffer = AttrTree()
            for task in self._scheduled_tasks:
                if isinstance(task, Analyze) and task.stackwise:
                    task(attrtree, self.time_fn(), times)
                else:
                    task(attrtree_buffer, self.time_fn(), times)
                    attrtree.update(attrtree_buffer)

        (self.fixed, attrtree.fixed) = (True, True)
        return attrtree


    def verify_times(self, times, strict=False):
        """
        Checks that tasks scheduled to be run at certain times will
        actually be executed. The strict flag determines whether to
        simply warn or raise an Exception.
        """
        for path, task in self.path_items.items():
            if task.times:
                unsatisfied = set(task.times) - set(times)
                if unsatisfied:
                    msg = "Task %r has been requested for times %s, " \
                          "not scheduled for collection." % (task, list(unsatisfied))

                if unsatisfied:
                    if strict: raise Exception(msg)
                    else:      param.main.warning(msg)


    def _schedule_tasks(self):
        """
        Inspect the path_items to find all the Collects that have
        been specified and add them to the scheduled tasks list.
        """
        self._scheduled_tasks = []
        for path, task in self.path_items.items():

            if not isinstance(task, Collect):
                self._scheduled_tasks = []
                raise Exception("Only Collects or Analyze objects allowed, not %s" % task)

            if isinstance(path, tuple) and task.mode == 'merge':
                self._scheduled_tasks = []
                raise Exception("Setting path for Task that is in 'merge' mode.")
            task.path = path
            self._scheduled_tasks.append(task)


    def __repr__(self):
        spec_strs = []
        for path, val in self.path_items.items():
            key = repr('.'.join(path)) if isinstance(path, tuple) else 'None'
            spec_strs.append('\n(%s, %r)' % (key, val))
        return 'Collector([%s])' % ', '.join(spec_strs)

    def __str__(self):
        indent = '  '
        num_items = len(self.path_items)
        padding = len(str(num_items))
        num_fmt = '%%0%dd.' % padding

        lines = ["%d tasks scheduled:\n" % num_items]
        dotted_line = indent + num_fmt +"  %s"
        merge_line = indent + num_fmt + "  [...] "
        value_line = indent*3 + ' '*padding + " %s %s"
        for i, (path, val) in enumerate(self.path_items.items()):
            if isinstance(path, tuple):
                lines.append(dotted_line % (i+1, '.'.join(p for p in path)))
                lines.append(value_line % (' ', val))
            else:
                lines.append(merge_line % (i+1))
                lines.append(value_line % ('', val))
        return '\n'.join(lines)


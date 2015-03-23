"""
Options and OptionTrees allow different classes of options
(e.g. matplotlib-specific styles and plot specific parameters) to be
defined separately from the core data structures and away from
visualization specific code.

There are three classes that form the options system:

Cycle:

   Used to define infinite cycles over a finite set of elements, using
   either an explicit list or the matplotlib rcParams. For instance, a
   Cycle object can be used loop a set of display colors for multiple
   curves on a single axis.

Options:

   Containers of arbitrary keyword values, including optional keyword
   validation, support for Cycle objects and inheritance.

OptionTree:

   A subclass of AttrTree that is used to define the inheritance
   relationships between a collection of Options objects. Each node
   of the tree supports a group of Options objects and the leaf nodes
   inherit their keyword values from parent nodes up to the root.

Store:

   A singleton class that stores all global and custom options and
   links HoloViews objects, the chosen plotting backend and the IPython
   extension together.

"""
import pickle
from contextlib import contextmanager

import numpy as np

import param
from .tree import AttrTree
from .util import sanitize_identifier
from .pprint import InfoPrinter

class OptionError(Exception):
    """
    Custom exception raised when there is an attempt to apply invalid
    options. Stores the necessary information to construct a more
    readable message for the user if caught and processed
    appropriately.
    """
    def __init__(self, invalid_keyword, allowed_keywords,
                 group_name=None, path=None):
        super(OptionError, self).__init__(self.message(invalid_keyword,
                                                       allowed_keywords,
                                                       group_name, path))
        self.invalid_keyword = invalid_keyword
        self.allowed_keywords = allowed_keywords
        self.group_name =group_name
        self.path = path


    def message(self, invalid_keyword, allowed_keywords, group_name, path):
        msg = ("Invalid option %s, valid options are: %s"
               % (repr(invalid_keyword), str(allowed_keywords)))
        if path and group_name:
            msg = ("Invalid key for group %r on path %r;\n"
                    % (group_name, path)) + msg
        return msg


class Cycle(param.Parameterized):
    """
    A simple container class that specifies cyclic options. A typical
    example would be to cycle the curve colors in an Overlay composed
    of an arbitrary number of curves. The values may be supplied as
    an explicit list or a key to look up in the default cycles
    attribute.
    """

    key = param.String(default='grayscale', doc="""
       Palettes look up the Palette values based on some key.""")

    values = param.List(default=[], doc="""
       The values the cycle will iterate over.""")

    default_cycles = {}

    def __init__(self, **params):
        super(Cycle, self).__init__(**params)
        self.values = self._get_values()


    def __getitem__(self, num):
        return self(values=self.values[:num])


    def _get_values(self):
        if self.values: return self.values
        elif self.key:
            return self.default_cycles[self.key]
        else:
            raise ValueError("Supply either a key or explicit values.")


    def __call__(self, values=None, **params):
        values = values if values else self.values
        return self.__class__(**dict(self.get_param_values(), values=values, **params))


    def __len__(self):
        return len(self.values)


    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, self.values)



def grayscale(val):
    return (val, val, val, 1.0)


class Palette(Cycle):
    """
    Palettes allow easy specifying a discrete sampling
    of an existing colormap. Palettes may be supplied a key
    to look up a function function in the colormap class
    attribute. The function should accept a float scalar
    in the specified range and return a RGB(A) tuple.
    The number of samples may also be specified as a
    parameter.

    The range and samples may conveniently be overridden
    with the __getitem__ method.
    """

    range = param.NumericTuple(default=(0, 1), doc="""
        The range from which the Palette values are sampled.""")

    samples = param.Integer(default=32, doc="""
        The number of samples in the given range to supply to
        the sample_fn.""")

    sample_fn = param.Callable(default=np.linspace, doc="""
        The function to generate the samples, by default linear.""")

    reverse = param.Boolean(default=False, doc="""
        Whether to reverse the palette.""")

    # A list of available colormaps
    colormaps = {'grayscale': grayscale}

    def __init__(self, key, **params):
        super(Cycle, self).__init__(key=key, **params)
        self.values = self._get_values()


    def __getitem__(self, slc):
        """
        Provides a convenient interface to override the
        range and samples parameters of the Cycle.
        Supplying a slice step or index overrides the
        number of samples. Unsupplied slice values will be
        inherited.
        """
        (start, stop), step = self.range, self.samples
        if isinstance(slc, slice):
            if slc.start is not None:
                start = slc.start
            if slc.stop is not None:
                stop = slc.stop
            if slc.step is not None:
                step = slc.step
        else:
            step = slc
        return self(range=(start, stop), samples=step)


    def _get_values(self):
        cmap = self.colormaps[self.key]
        (start, stop), steps = self.range, self.samples
        samples = [cmap(n) for n in self.sample_fn(start, stop, steps)]
        return samples[::-1] if self.reverse else samples



class Options(param.Parameterized):
    """
    An Options object holds a collection of keyword options. In
    addition, Options support (optional) keyword validation as well as
    infinite indexing over the set of supplied cyclic values.

    Options support inheritance of setting values via the __call__
    method. By calling an Options object with additional keywords, you
    can create a new Options object inheriting the parent options.
    """

    allowed_keywords = param.List(default=None, allow_None=True, doc="""
       Optional list of strings corresponding to the allowed keywords.""")

    key = param.String(default=None, allow_None=True, doc="""
       Optional specification of the options key name. For instance,
       key could be 'plot' or 'style'.""")


    def __init__(self, key=None, allowed_keywords=None, **kwargs):
        for kwarg in sorted(kwargs.keys()):
            if allowed_keywords and kwarg not in allowed_keywords:
                raise OptionError(kwarg, allowed_keywords)

        self.kwargs = kwargs
        self._options = self._expand_options(kwargs)
        allowed_keywords = sorted(allowed_keywords) if allowed_keywords else None
        super(Options, self).__init__(allowed_keywords=allowed_keywords, key=key)


    def __call__(self, allowed_keywords=None, **kwargs):
        """
        Create a new Options object that inherits the parent options.
        """
        allowed_keywords=self.allowed_keywords if allowed_keywords is None else allowed_keywords
        inherited_style = dict(allowed_keywords=allowed_keywords, **kwargs)
        return self.__class__(key=self.key, **dict(self.kwargs, **inherited_style))


    def _expand_options(self, kwargs):
        """
        Expand out Cycle objects into multiple sets of keyword values.

        To elaborate, the full Cartesian product over the supplied
        Cycle objects is expanded into a list, allowing infinite,
        cyclic indexing in the __getitem__ method."""
        filter_static = dict((k,v) for (k,v) in kwargs.items() if not isinstance(v, Cycle))
        filter_cycles = [(k,v) for (k,v) in kwargs.items() if isinstance(v, Cycle)]

        if not filter_cycles: return [kwargs]

        filter_names, filter_values = list(zip(*filter_cycles))

        cyclic_tuples = list(zip(*[val.values for val in filter_values]))
        return [dict(zip(filter_names, tps), **filter_static) for tps in cyclic_tuples]


    def keys(self):
        "The keyword names across the supplied options."
        return sorted(list(self.kwargs.keys()))


    def max_cycles(self, num):
        """
        Truncates all contained Cycle objects to a maximum number
        of Cycles and returns a new Options object with the
        truncated or resampled Cycles.
        """
        kwargs = {kw: (arg[num] if isinstance(arg, Cycle) else arg)
                  for kw, arg in self.kwargs.items()}
        return self(**kwargs)


    def __getitem__(self, index):
        """
        Infinite cyclic indexing of options over the integers,
        looping over the set of defined Cycle objects.
        """
        return dict(self._options[index % len(self._options)])


    @property
    def options(self):
        "Access of the options keywords when no cycles are defined."
        if len(self._options) == 1:
            return dict(self._options[0])
        else:
            raise Exception("The options property may only be used with non-cyclic Options.")


    def __repr__(self):
        kws = ', '.join("%s=%r" % (k,v) for (k,v) in self.kwargs.items())
        return "%s(%s)" % (self.__class__.__name__,  kws)

    def __str__(self):
        return repr(self)



class OptionTree(AttrTree):
    """
    A subclass of AttrTree that is used to define the inheritance
    relationships between a collection of Options objects. Each node
    of the tree supports a group of Options objects and the leaf nodes
    inherit their keyword values from parent nodes up to the root.

    Supports the ability to search the tree for the closest valid path
    using the find method, or compute the appropriate Options value
    given an object and a mode. For a given node of the tree, the
    options method computes a Options object containing the result of
    inheritance for a given group up to the root of the tree.
    """

    def __init__(self, items=None, identifier=None, parent=None, groups=None):
        if groups is None:
            raise ValueError('Please supply groups dictionary')
        self.__dict__['groups'] = groups
        self.__dict__['_instantiated'] = False
        AttrTree.__init__(self, items, identifier, parent)
        self.__dict__['_instantiated'] = True


    def _inherited_options(self, identifier, group_name, options):
        """
        Computes the inherited Options object for the given group
        name from the current node given a new set of options.
        """
        override_kwargs = dict(options.kwargs)
        if not self._instantiated:
            override_kwargs['allowed_keywords'] = options.allowed_keywords
        elif identifier in self.children:
            override_kwargs['allowed_keywords'] = self[identifier][group_name].allowed_keywords

        if group_name not in self.groups:
            raise KeyError("Group %s not defined on SettingTree" % group_name)

        current_node = self[identifier] if identifier in self.children else self
        group_options = current_node.groups[group_name]
        try:
            return group_options(**override_kwargs)
        except OptionError as e:
            raise OptionError(e.invalid_keyword,
                              e.allowed_keywords,
                              group_name=group_name,
                              path = self.path)


    def __getitem__(self, item):
        if item in self.groups:
            return self.groups[item]
        return super(OptionTree, self).__getitem__(item)


    def __getattr__(self, identifier):
        """
        Allows creating sub OptionTree instances using attribute
        access, inheriting the group options.
        """
        try:
            return super(AttrTree, self).__getattr__(identifier)
        except AttributeError: pass

        if identifier.startswith('_'):   raise AttributeError(str(identifier))
        elif self.fixed==True:           raise AttributeError(self._fixed_error % identifier)

        valid_id = sanitize_identifier(identifier, escape=False)
        if valid_id in self.children:
            return self.__dict__[valid_id]

        self.__setattr__(identifier, self.groups)
        return self[identifier]


    def __setattr__(self, identifier, val):
        identifier = sanitize_identifier(identifier, escape=False)
        new_groups = {}
        if isinstance(val, dict):
            group_items = val
        elif isinstance(val, Options) and val.key is None:
            raise AttributeError("Options object needs to have a group name specified.")
        elif isinstance(val, Options):
            group_items = {val.key: val}
        elif isinstance(val, OptionTree):
            group_items = val.groups

        current_node = self[identifier] if identifier in self.children else self
        for group_name in current_node.groups:
            options = group_items.get(group_name, False)
            if options:
                new_groups[group_name] = self._inherited_options(identifier, group_name, options)
            else:
                new_groups[group_name] = current_node.groups[group_name]

        if new_groups:
            new_node = OptionTree(None, identifier=identifier, parent=self, groups=new_groups)
        else:
            raise ValueError('OptionTree only accepts a dictionary of Options.')
        super(OptionTree, self).__setattr__(identifier, new_node)

        if isinstance(val, OptionTree):
            for subtree in val:
                self[identifier].__setattr__(subtree.identifier, subtree)


    def find(self, path, mode='node'):
        """
        Find the closest node or path to an the arbitrary path that is
        supplied down the tree from the given node. The mode argument
        may be either 'node' or 'path' which determines the return
        type.
        """
        path = path.split('.') if isinstance(path, str) else list(path)
        item = self

        for idx, child in enumerate(path):
            escaped_child = sanitize_identifier(child, escape=False)
            matching_children = [c for c in item.children
                                 if child.endswith(c) or escaped_child.endswith(c)]
            matching_children = sorted(matching_children, key=lambda x: -len(x))
            if matching_children:
                item = item[matching_children[0]]
            else:
                continue
        return item if mode == 'node' else item.path


    def closest(self, obj, group):
        """
        This method is designed to be called from the root of the
        tree. Given any LabelledData object, this method will return
        the most appropriate Options object, including inheritance.

        In addition, closest supports custom options by checking the
        object
        """
        components = (obj.__class__.__name__, obj.group, obj.label)
        return self.find(components).options(group)



    def options(self, group, target=None):
        """
        Using inheritance up to the root, get the complete Options
        object for the given node and the specified group.
        """
        if target is None:
            target = self.path
        if self.groups.get(group, None) is None:
            return None
        if self.parent is None and target and (self is not Store.options):
            root_name = self.__class__.__name__
            replacement = root_name + ('' if len(target) == len(root_name) else '.')
            option_key = target.replace(replacement,'')
            if option_key in Store.options:
                return Store.options[option_key].options(group)
            else:
                return Options()
        elif self.parent is None:
            return self.groups[group]

        return Options(**dict(self.parent.options(group,target=target).kwargs,
                              **self.groups[group].kwargs))


    def _node_identifier(self, node):
        if node.parent is None:
            return '--+'
        else:
            values = ', '.join([repr(group) for group in node.groups.values()])
            return "%s: %s" % (node.identifier, values)


    def __repr__(self):
        if len(self) == 0:
            return self._node_identifier(self)
        return super(OptionTree, self).__repr__()



class Compositor(param.Parameterized):
    """
    A Compositor is a way of specifying an operation to be automatically
    applied to Overlays that match a specified pattern upon display.

    Any ElementOperation that takes an Overlay as input may be used to
    define a compositor.

    For instance, a compositor may be defined to automatically display
    three overlaid monochrome matrices as an RGB image as long as the
    values names of those matrices match 'R', 'G' and 'B'.
    """

    mode = param.ObjectSelector(default='data',
                                objects=['data', 'display'], doc="""
      The mode of the Compositor object which may be either 'data' or
      'display'.""")

    operation = param.Parameter(doc="""
       The ElementOperation to apply when collapsing overlays.""")

    pattern = param.String(doc="""
       The overlay pattern to be processed. An overlay pattern is a
       sequence of elements specified by dotted paths separated by * .

       For instance the following pattern specifies three overlayed
       matrices with values of 'RedChannel', 'GreenChannel' and
       'BlueChannel' respectively:

      'Image.RedChannel * Image.GreenChannel * Image.BlueChannel.

      This pattern specification could then be associated with the RGB
      operation that returns a single RGB matrix for display.""")

    group = param.String(doc="""
       The group identifier for the output of this particular compositor""")

    kwargs = param.Dict(doc="""
       Optional set of parameters to pass to the operation.""")


    operations = []  # The operations that can be used to define compositors.
    definitions = [] # The set of all the compositor instances


    @classmethod
    def strongest_match(cls, overlay, mode):
        """
        Returns the single strongest matching compositor operation
        given an overlay. If no matches are found, None is returned.

        The best match is defined as the compositor operation with the
        highest match value as returned by the match_level method.
        """
        match_strength = [(op.match_level(overlay), op) for op in cls.definitions
                          if op.mode == mode]
        matches = [(match[0], op, match[1]) for (match, op) in match_strength if match is not None]
        if matches == []: return None
        else:             return sorted(matches)[0]


    @classmethod
    def collapse_element(cls, overlay, key=None, ranges=None, mode='data'):
        """
        Finds any applicable compositor and applies it.
        """
        from .overlay import Overlay
        match = cls.strongest_match(overlay, mode)
        if match is None: return overlay
        (_, applicable_op, (start, stop)) = match
        values = overlay.values()
        sliced = Overlay.from_values(values[start:stop])
        result = applicable_op.apply(sliced, ranges, key=key)
        result = result.relabel(group=applicable_op.group)
        output = Overlay.from_values(values[:start]+[result]+values[stop:])
        output.id = overlay.id
        return output


    @classmethod
    def collapse(cls, holomap, ranges=None, mode='data'):
        """
        Given a map of Overlays, apply all applicable compositors.
        """
        # No potential compositors
        if cls.definitions == []:
            return holomap

        # Apply compositors
        clone = holomap.clone(shared_data=False)
        data = zip(ranges[1], holomap.data.values()) if ranges else holomap.data.items()
        for key, overlay in data:
            clone[key] = cls.collapse_element(overlay, key, ranges, mode)
        return clone

    @classmethod
    def register(cls, compositor):
        defined_groups = [op.group for op in cls.definitions]
        if compositor.group in defined_groups:
            cls.definitions.pop(defined_groups.index(compositor.group))
        cls.definitions.append(compositor)
        if compositor.operation not in cls.operations:
            cls.operations.append(compositor.operation)


    def __init__(self, pattern, operation, group, mode, **kwargs):
        self._pattern_spec, labels = [], []

        for path in pattern.split('*'):
            path_tuple = tuple(el.strip() for el in path.strip().split('.'))
            self._pattern_spec.append(path_tuple)

            if len(path_tuple) == 3:
                labels.append(path_tuple[2])

        if len(labels) > 1 and not all(l==labels[0] for l in labels):
            raise KeyError("Mismatched labels not allowed in compositor patterns")
        elif len(labels) == 1:
            self.label = labels[0]
        else:
            self.label = ''

        super(Compositor, self).__init__(group=group,
                                         pattern=pattern,
                                         operation=operation,
                                         mode=mode,
                                         kwargs=kwargs)


    @property
    def output_type(self):
        """
        Returns the operation output_type unless explicitly overridden
        in the kwargs.
        """
        if 'output_type' in self.kwargs:
            return self.kwargs['output_type']
        else:
            return self.operation.output_type


    def _slice_match_level(self, overlay_items):
        """
        Find the match strength for a list of overlay items that must
        be exactly the same length as the pattern specification.
        """
        level = 0
        for spec, el in zip(self._pattern_spec, overlay_items):
            if spec[0] != type(el).__name__:
                return None
            level += 1      # Types match
            if len(spec) == 1: continue

            group = [el.group, sanitize_identifier(el.group, escape=False)]
            if spec[1] in group: level += 1  # Values match
            else:                     return None

            if len(spec) == 3:
                group = [el.label, sanitize_identifier(el.label, escape=False)]
                if (spec[2] in group):
                    level += 1  # Labels match
                else:
                    return None
        return level


    def match_level(self, overlay):
        """
        Given an overlay, return the match level and applicable slice
        of the overall overlay. The level an integer if there is a
        match or None if there is no match.

        The level integer is the number of matching components. Higher
        values indicate a stronger match.
        """
        slice_width = len(self._pattern_spec)
        if slice_width > len(overlay): return None

        # Check all the possible slices and return the best matching one
        best_lvl, match_slice = (0, None)
        for i in range(len(overlay)-slice_width+1):
            overlay_slice = overlay.values()[i:i+slice_width]
            lvl = self._slice_match_level(overlay_slice)
            if lvl is None: continue
            if lvl > best_lvl:
                best_lvl = lvl
                match_slice = (i, i+slice_width)

        return (best_lvl, match_slice) if best_lvl != 0 else None


    def apply(self, value, input_ranges, key=None):
        """
        Apply the compositor on the input with the given input ranges.
        """
        from .overlay import CompositeOverlay
        if isinstance(value, CompositeOverlay) and len(value) == 1:
            value = value.values()[0]
        if key is None:
            return self.operation(value, input_ranges=input_ranges, **self.kwargs)
        return self.operation.instance(input_ranges=input_ranges, **self.kwargs).process_element(value, key)



class Store(object):
    """
    The Store is what links up HoloViews objects to their
    corresponding options and to the appropriate classes of the chosen
    backend (e.g for rendering).

    In addition, Storesupports pickle operations that automatically
    pickle and unpickle the corresponding options for a HoloViews
    object.
    """

    # A mapping from ViewableElement types to their corresponding plot
    # types. Set using the register_plots methods.
    registry = {}

    # Once register_plotting_classes is called, this OptionTree is populated
    options = OptionTree(groups={'plot':  Options(),
                                 'style': Options(),
                                 'norm':  Options()})

    # A dictionary of custom OptionTree by custom object id
    custom_options = {}
    load_counter_offset = None
    save_option_state = False

    @classmethod
    def load(cls, filename):
        """
        Equivalent to pickle.load except that the HoloViews trees is
        restored appropriately.
        """
        cls.load_counter_offset = max(cls.custom_options) if cls.custom_options else 0
        val = pickle.load(filename)
        cls.load_counter_offset = None
        return val


    @classmethod
    def info(cls, obj, ansi=True):
        """
        Show information about a particular object or component class
        including the applicable style and plot options.
        """
        print(InfoPrinter.info(obj, ansi=ansi))

    @classmethod
    def loads(cls, pickle_string):
        """
        Equivalent to pickle.loads except that the HoloViews trees is
        restored appropriately.
        """
        cls.load_counter_offset = max(cls.custom_options) if cls.custom_options else 0
        val = pickle.loads(pickle_string)
        cls.load_counter_offset = None
        return val


    @classmethod
    def dump(cls, obj, filename, protocol=0):
        """
        Equivalent to pickle.dump except that the HoloViews option
        tree is saved appropriately.
        """
        cls.save_option_state = True
        pickle.dump(obj, filename, protocol=protocol)
        cls.save_option_state = False

    @classmethod
    def dumps(cls, obj, protocol=0):
        """
        Equivalent to pickle.dumps except that the HoloViews option
        tree is saved appropriately.
        """
        cls.save_option_state = True
        val = pickle.dumps(obj, protocol=protocol)
        cls.save_option_state = False
        return val


    @classmethod
    def lookup_options(cls, obj, group):
        if obj.id is None:
            return cls.options.closest(obj, group)
        elif obj.id in cls.custom_options:
            return cls.custom_options[obj.id].closest(obj, group)
        else:
            raise KeyError("No custom settings defined for object with id %d" % obj.id)

    @classmethod
    def lookup(cls, obj):
        """
        Given an object, lookup the corresponding customized option
        tree if a single custom tree is applicable.
        """
        ids = set([el for el in obj.traverse(lambda x: x.id) if el is not None])
        if len(ids) == 0:
            raise Exception("Object does not own a custom options tree")
        elif len(ids) != 1:
            idlist = ",".join([str(el) for el in sorted(ids)])
            raise Exception("Object contains elements combined across "
                            "multiple custom trees (ids %s)" % idlist)
        return cls.custom_options[list(ids)[0]]


    @classmethod
    def add_style_opts(cls, component, new_options):
        """
        Given a component such as an Element (e.g. Image, Curve) or a
        container (e.g Layout) specify new style options to be
        accepted by the corresponding plotting class.

        Note: This is supplied for advanced users who know which
        additional style keywords are appropriate for the
        corresponding plotting class.
        """
        if component not in cls.registry:
            raise ValueError("Component %r not registered to a plotting class" % component)

        if not isinstance(new_options, list) or not all(isinstance(el, str) for el in new_options):
            raise ValueError("Please supply a list of style option keyword strings")

        with param.logging_level('CRITICAL'):
            for option in new_options:
                if option not in cls.registry[component].style_opts:
                    cls.registry[component].style_opts.append(option)
        cls.register_plots()


    @classmethod
    def register_plots(cls):
        """
        Given that the Store.registry dictionary has been populate
        with {<element>:<plot-class>} items, build an OptionTree for the
        supported plot types, registering allowed plotting and style
        keywords.

        This is designed to be backend independent but makes the
        following assumptions:

        * Plotting classes are param.Parameterized objects.

        * Plotting classes have a style_opts list of keywords used to
          control the display style of the output.

        * Overlay plotting is a function of the overlaid elements and
          only has plot options (and not style or normalization
          options).
        """
        from .overlay import CompositeOverlay
        path_items = {}
        for view_class, plot in cls.registry.items():
            name = view_class.__name__
            plot_opts = [k for k in plot.params().keys() if k not in ['name']]
            style_opts = plot.style_opts
            opt_groups = {'plot': Options(allowed_keywords=plot_opts)}

            if not isinstance(view_class, CompositeOverlay) or hasattr(plot, 'style_opts'):
                opt_groups.update({'style': Options(allowed_keywords=style_opts),
                                   'norm':  Options(framewise=False, axiswise=False,
                                                    allowed_keywords=['framewise',
                                                                      'axiswise'])})
            path_items[name] = opt_groups

        cls.options = OptionTree(sorted(path_items.items()),
                                  groups={'style': Options(),
                                          'plot': Options(),
                                          'norm': Options()})

class StoreOptions(object):
    """
    A collection of utilities for advanced users for creating and
    setting customized option tress on the Store. Designed for use by
    either advanced users or the %opts line and cell magics which use
    this machinery.
    """

    #===============#
    # ID management #
    #===============#

    @classmethod
    def get_object_ids(cls, obj):
        return set(el for el
                   in obj.traverse(lambda x: getattr(x, 'id', None))
                   if el is not None)


    @classmethod
    def propagate_ids(cls, obj, new_id, applied_keys):
        """
        Recursively propagate an id through an object for components
        matching the applied_keys. This method can only be called if
        there is a tree with a matching id in Store.custom_options
        """
        if not new_id in Store.custom_options:
            raise AssertionError("The set_ids method requires "
                                 "Store.custom_options to contain"
                                 " a tree with id %d" % new_id)
        obj.traverse(lambda o: setattr(o, 'id', new_id), specs=set(applied_keys))

    @classmethod
    def capture_ids(cls, obj):
        """
        Given an list of ids, capture a list of ids that can be
        restored using the restore_ids.
        """
        return obj.traverse(lambda o: getattr(o, 'id'))

    @classmethod
    def restore_ids(cls, obj, ids):
        """
        Given an list of ids as captured with capture_ids, restore the
        ids. Note the structure of an object must not change between
        the calls to capture_ids and restore_ids.
        """
        ids = iter(ids)
        obj.traverse(lambda o: setattr(o, 'id', next(ids)))


    @classmethod
    def apply_customizations(cls, spec, options):
        """
        Apply the given option specs to the supplied options tree.
        """
        for key in sorted(spec.keys()):
            if isinstance(spec[key], (list, tuple)):
                customization = {v.key:v for v in spec[key]}
            else:
                customization = {k:(Options(**v) if isinstance(v, dict) else v)
                                 for k,v in spec[key].items()}
            options[str(key)] = customization
        return options


    @classmethod
    def validate_spec(cls, spec):
        """
        Given a specification, validated it against the default
        options tree (Store.options)
        """
        options = OptionTree(items=Store.options.data.items(),
                             groups=Store.options.groups)
        return cls.apply_customizations(spec, options)


    @classmethod
    def expand_compositor_keys(cls, spec):
        """
        Expands compositor definition keys into {type}.{group}
        keys. For instance a compositor operation returning a group
        string 'Image' of element type RGB expands to 'RGB.Image'.
        """
        expanded_spec={}
        applied_keys = []
        compositor_defs = {el.group:el.output_type.__name__
                           for el in Compositor.definitions}
        for key, val in spec.items():
            if key not in compositor_defs:
                expanded_spec[key] = val
            else:
                # Send id to Overlays
                applied_keys = ['Overlay']
                type_name = compositor_defs[key]
                expanded_spec[str(type_name+'.'+key)] = val
        return expanded_spec, applied_keys


    @classmethod
    def create_custom_trees(cls, obj, options=None):
        """
        Returns the appropriate set of customized subtree clones for
        an object, suitable for merging with Store.custom_options (i.e
        with the ids appropriately offset). Note if an object has no
        integer ids a new OptionTree is built.
        """
        clones = {}
        obj_ids = cls.get_object_ids(obj)
        store_ids = Store.custom_options.keys()
        offset = (max(store_ids)+1) if len(store_ids) > 0 else 0
        for tree_id in obj_ids:
            original = Store.custom_options[tree_id]
            clone = OptionTree(items=  original.items(),
                               groups= original.groups)
            clones[tree_id + offset + 1] = clone

        if len(clones) == 0:
            new_tree = OptionTree(groups={'norm': Options(),
                                          'plot': Options(),
                                          'style': Options()})
            clones[offset] = new_tree

        return {k:cls.apply_customizations(options, t) if options else t
                for k,t in clones.items()}


    @classmethod
    def merge_options(cls, options=None, **kwargs):
        """
        Given a full options dictionary and options groups specified
        as a keywords such as return the full set of merged options:

        >>> options={'Curve':{'style':dict(color='b')}}
        >>> style={'Curve':{'linewidth':10 }}
        >>> merged = StoreOptions.merge_options(options, style=style)
        >>> sorted(merged['Curve']['style'].items())
        [('color', 'b'), ('linewidth', 10)]
        """
        options = {} if (options is None) else dict(**options)
        all_keys = set(k for d in kwargs.values() for k in d)
        for spec_key in all_keys:
            additions = {}
            for k, d in kwargs.items():
                if spec_key in d:
                    kws = d[spec_key]
                    additions.update({k:kws})
            if spec_key not in options:
                options[spec_key] = {}
            for key in additions:
                if key in options[spec_key]:
                    options[spec_key][key].update(additions[key])
                else:
                    options[spec_key][key] = additions[key]
        return options


    @classmethod
    @contextmanager
    def options(cls, obj, options=None, **kwargs):
        """
        Context-manager for temporarily setting options on an object
        (if options is None, no options will be set) . Once the
        context manager exits, both the object and the Store will be
        left in exactly the same state they were in before the context
        manager was used.

        See holoviews.core.options.set_options function for more
        information on the options specification format.
        """
        if (options is None) and kwargs == {}: yield
        else:
            options = cls.merge_options(options, **kwargs)
            ids = cls.capture_ids(obj)
            original_custom_keys = set(Store.custom_options.keys())
            cls.set_options(obj, options)
            yield
        if options is not None:
            current_custom_keys = set(Store.custom_options.keys())
            for key in current_custom_keys.difference(original_custom_keys):
                del Store.custom_options[key]
                cls.restore_ids(obj, ids)


    @classmethod
    def set_options(cls, obj, options=None, **kwargs):
        """
        Pure Python function for customize HoloViews objects in terms of
        their style, plot and normalization options.

        The options specification is a dictionary containing the target
        for customization as a {type}.{group}.{label} keys. An example of
        such a key is 'Image' which would customize all Image components
        in the object. The key 'Image.Channel' would only customize Images
        in the object that have the group 'Channel'.

        The corresponding value is then a list of Option objects specified
        with an appropriate category ('plot', 'style' or 'norm'). For
        instance, using the keys described above, the specs could be:

        {'Image:[Options('style', cmap='jet')]}

        Or setting two types of option at once:

        {'Image.Channel:[Options('plot', size=50),
                        Options('style', cmap='Blues')]}


        Relationship to the %%opts magic
        ----------------------------------

        This function matches the functionality supplied by the %%opts
        cell magic in the IPython extension. In fact, you can use the same
        syntax as the IPython cell magic to achieve the same customization
        as shown above:

        from holoviews.ipython.parser import OptsSpec
        set_options(my_image, OptsSpec.parse("Image (cmap='jet')"))

        Then setting both plot and style options:

        set_options(my_image, OptsSpec.parse("Image [size=50] (cmap='Blues')"))
        """
        # Note that an alternate, more verbose and less recommended
        # syntax can also be used:

        # {'Image.Channel:{'plot':  Options(size=50),
        #                  'style': Options('style', cmap='Blues')]}
        options = cls.merge_options(options, **kwargs)
        spec, compositor_applied = StoreOptions.expand_compositor_keys(options)

        custom_trees = StoreOptions.create_custom_trees(obj, spec)
        Store.custom_options.update(custom_trees)
        for tree_id in custom_trees.keys():
            cls.propagate_ids(obj, tree_id, compositor_applied+list(spec.keys()))
        return obj

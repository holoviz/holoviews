"""
Options and OptionTrees allow different classes of options
(e.g. matplotlib specific styles and plot specific parameters) to be
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

"""

import param
from .tree import AttrTree
from .operation import ElementOperation


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
    of an arbitrary number of curves.

    A Cycles object accepts either a list of items to cycle over or an
    rckey string used to look up the elements in the matplotlib
    rcParams dictionary.
    """

    items = param.List(default=None, allow_None=True,  doc="""
        If supplied, the explicit list of items to be cycled over.""")

    rckey = param.String(default='axes.color_cycle', doc="""
       If elements is None, this is the key in the matplotlib rcParams
       to use to get the cycle elements""")

    def __init__(self, items=None, **params):
        super(Cycle, self).__init__(items=items, **params)

    @property
    def elements(self):
        if self.items is None:
            from matplotlib import rcParams
            return rcParams[self.rckey]
        else:
            return self.items


    def __len__(self):
        return len(self.elements)


    def __repr__(self):
        return "Cycle(%s)" % self.elements



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

        allowed_keywords = sorted(allowed_keywords) if allowed_keywords else None
        super(Options, self).__init__(allowed_keywords=allowed_keywords, key=key)

        for kwarg in kwargs:
            if allowed_keywords and kwarg not in allowed_keywords:
                raise OptionError(kwarg, allowed_keywords)
        self.kwargs = kwargs
        self._options = self._expand_options(kwargs)


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
        if not all(len(c)==len(filter_values[0]) for c in filter_values):
            raise Exception("Cycle objects supplied with different lengths")

        cyclic_tuples = list(zip(*[val.elements for val in filter_values]))
        return [dict(zip(filter_names, tps), **filter_static) for tps in cyclic_tuples]


    def keys(self):
        "The keyword names across the supplied options."
        return sorted(list(self.kwargs.keys()))


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
        override_kwargs = options.kwargs
        if not self._instantiated:
            override_kwargs['allowed_keywords'] = options.allowed_keywords
        elif identifier in self.children:
            override_kwargs['allowed_keywords'] = self[identifier][group_name].allowed_keywords

        if group_name not in self.groups:
            raise KeyError("Group %s not defined on SettingTree" % group_name)

        group_options = self.groups[group_name]
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


    def __setattr__(self, identifier, val):
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
            matching_children = [c for c in item.children if child.endswith(c)]
            matching_children = sorted(matching_children, key=lambda x: len(x))
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
        components = (obj.__class__.__name__, obj.value, obj.label)
        return self.find(components).options(group)


    def options(self, group):
        """
        Using inheritance up to the root, get the complete Options
        object for the given node and the specified group.
        """
        if self.groups.get(group, None) is None:
            return None
        if self.parent is None:
            return self.groups[group]
        return Options(**dict(self.parent.options(group).kwargs,
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



class ChannelDefinition(param.Parameterized):
    """
    A ChannelDefinition object defines an operation to be
    automatically collapse certain channels within Overlays. For
    instance, this can be used to display three overlaid monochrome
    matrices as an RGB image.
    """

    operation = param.ClassSelector(class_=ElementOperation, is_instance=False, doc="""
       The ElementOperation to apply when combining channels""")

    pattern = param.String(doc="""
       The overlay pattern to be processed. An overlay pattern is a
       sequence of elements specified by dotted paths separated by * .

       For instance the following pattern specifies three overlayed
       matrices with values of 'RedChannel', 'GreenChannel' and
       'BlueChannel' respectively:

      'Matrix.RedChannel * Matrix.GreenChannel * Matrix.BlueChannel.

      This pattern specification could then be associated with the RGB
      operation that returns a single RGB matrix for display.""")

    value = param.String(doc="""
       The value identifier for the output of this particular
       ChannelDefinition definition.""")

    kwargs = param.Dict(doc="""
       Optional set of parameters to pass to the operation.""")

    operations = []

    definitions = []

    def __init__(self, pattern, operation, value, **kwargs):
        if not any (operation is op for op in self.operations):
            raise ValueError("Operation %r not in allowed operations" % operation)
        self._pattern_spec, labels = [], []

        for path in pattern.split('*'):
            path_tuple = tuple(el.strip() for el in path.strip().split('.'))

            if path_tuple[0] != 'Matrix':
                raise KeyError("Only Matrix is currently supported in channel operation patterns")

            self._pattern_spec.append(path_tuple)

            if len(path_tuple) == 3:
                labels.append(path_tuple[2])

        if len(labels) > 1 and not all(l==labels[0] for l in labels):
            raise KeyError("Mismatched labels not allowed in channel operation patterns")
        elif len(labels) == 1:
            self.label = labels[0]
        else:
            self.label = ''

        super(ChannelDefinition, self).__init__(value=value,
                                                pattern=pattern,
                                                operation=operation,
                                                kwargs=kwargs)


    @classmethod
    def _collapse(cls, overlay, pattern, fn, style_key):
        """
        Given an overlay object collapse the channels according to
        pattern using the supplied function. Any collapsed ViewableElement is
        then given the supplied style key.
        """
        pattern = [el.strip() for el in pattern.rsplit('*')]
        if len(pattern) > len(overlay): return overlay

        skip=0
        collapsed_overlay = overlay.clone(None)
        for i, key in enumerate(overlay.keys()):
            layer_labels = overlay.labels[i:len(pattern)+i]
            matching = all(l.endswith(p) for l, p in zip(layer_labels, pattern))
            if matching and len(layer_labels)==len(pattern):
                views = [el for el in overlay if el.label in layer_labels]
                if isinstance(overlay, Overlay):
                    views = np.product([Overlay.from_view(el) for el in overlay])
                overlay_slice = overlay.clone(views)
                collapsed_view = fn(overlay_slice)
                if isinstance(overlay, LayoutTree):
                    collapsed_overlay *= collapsed_view
                else:
                    collapsed_overlay[key] = collapsed_view
                skip = len(views)-1
            elif skip:
                skip = 0 if skip <= 0 else (skip - 1)
            else:
                if isinstance(overlay, LayoutTree):
                    collapsed_overlay *= overlay[key]
                else:
                    collapsed_overlay[key] = overlay[key]
        return collapsed_overlay


    @classmethod
    def collapse_channels(cls, vmap):
        """
        Given a map of Overlays, apply all applicable channel
        reductions.
        """
        if not issubclass(vmap.type, CompositeOverlay):
            return vmap
        elif not CompositeOverlay.channels.keys(): # No potential channel reductions
            return vmap

        # Apply all customized channel operations
        collapsed_vmap = vmap.clone()
        for key, overlay in vmap.items():
            customized = [k for k in CompositeOverlay.channels.keys()
                          if overlay.label and k.startswith(overlay.label)]
            # Largest reductions should be applied first
            sorted_customized = sorted(customized, key=lambda k: -CompositeOverlay.channels[k].size)
            sorted_reductions = sorted(CompositeOverlay.channels.options(),
                                       key=lambda k: -CompositeOverlay.channels[k].size)
            # Collapse the customized channel before the other definitions
            for key in sorted_customized + sorted_reductions:
                channel = CompositeOverlay.channels[key]
                if channel.mode is None: continue
                collapse_fn = channel.operation
                fn = collapse_fn.instance(**channel.opts)
                collapsed_vmap[k] = cls._collapse(overlay, channel.pattern, fn, key)
        return vmap

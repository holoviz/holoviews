"""
Settings and SettingsTrees allow different classes of settings
(e.g. matplotlib specific styles and plot specific parameters) to be
defined separately from the core data structures and away from
visualization specific code.

There are three classes that form the settings system:

Cycle:

   Used to define infinite cycles over a finite set of elements, using
   either an explicit list or the matplotlib rcParams. For instance, a
   Cycle object can be used loop a set of display colors for multiple
   curves on a single axis.

Settings:

   Containers of arbitrary keyword values, including optional keyword
   validation, support for Cycle objects and inheritance.

SettingsTree:

   A subclass of AttrTree that is used to define the inheritance
   relationships between a collection of Settings objects. Each node
   of the tree supports a group of Settings objects and the leaf nodes
   inherit their keyword values from parent nodes up to the root.

"""

from .tree import AttrTree


class Cycle(object):
    """
    A simple container class that specifies cyclic settings. A typical
    example would be to cycle the curve colors in an Overlay composed
    of an arbitrary number of curves.

    A Cycles object accepts either a list of elements or an rckey
    string used to look up the elements in the matplotlib rcParams.
    """

    def __init__(self, elements=[], rckey='axes.color_cycle'):
        self.rckey = rckey
        if len(elements):
            self._elements = elements
        else:
            self._elements = None


    @property
    def elements(self):
        if self._elements is None:
            from matplotlib import rcParams
            return rcParams[self.rckey]
        else:
            return self._elements


    def __len__(self):
        return len(self.elements)


    def __repr__(self):
        return "Cycle(%s)" % self.elements



class Settings(object):
    """
    A Settings object holds a collection of keyword options. In
    addition, Settings support (optional) keyword validation as well
    as infinite indexing over the set of supplied cyclic values.

    Settings support inheritance of setting values via the __call__
    method. By calling a Settings object with additional keywords, you
    can create a new Settings object inheriting the parent settings.

    valid_keywords: Optional list of strings corresponding to the allowed keywords.
    viewable_name:  The name of object that the settings apply to.
    **kwargs:       The keyword items corresponding to the stored settings.
    """

    def __init__(self, valid_keywords=None, viewable_name=None, **kwargs):
        self.valid_keywords = sorted(valid_keywords) if valid_keywords else None
        self.viewable_name = viewable_name
        self.kwargs = kwargs

        for kwarg in kwargs:
            if valid_keywords and kwarg not in valid_keywords:
                raise KeyError("Invalid option %s, valid settings for %s are: %s"
                               % (repr(kwarg), self.viewable_name, str(self.valid_keywords)))

        self._settings = self._expand_settings(kwargs)


    def __call__(self, valid_keywords=None, viewable_name=None, **kwargs):
        """
        Create a new Settings object that inherits the parent settings.
        """
        valid_keywords=self.valid_keywords if valid_keywords is None else valid_keywords
        viewable_name=self.viewable_name if viewable_name is None else viewable_name

        inherited_style = dict(valid_keywords=valid_keywords,
                               viewable_name=viewable_name, **kwargs)
        return self.__class__(**dict(self.kwargs, inherited_style))


    def _expand_settings(self, kwargs):
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
        "The keyword names across the supplied settings."
        return sorted(list(self.kwargs.keys()))


    def __getitem__(self, index):
        """
        Infinite cyclic indexing of settings over the integers,
        looping over the set of defined Cycle objects.
        """
        return dict(self._settings[index % len(self._settings)])


    @property
    def settings(self):
        "Access of the settings keywords when no cycles are defined."
        if len(self._settings) == 1:
            return dict(self._settings[0])
        else:
            raise Exception("The settings property may only be used with non-cyclic Settings.")


    def __repr__(self):
        kws = ', '.join("%s=%r" % (k,v) for (k,v) in self.kwargs.items())
        return "%s(%s)" % (self.__class__.__name__,  kws)



class SettingsTree(AttrTree):
    """
    A subclass of AttrTree that is used to define the inheritance
    relationships between a collection of Settings objects. Each node
    of the tree supports a group of Settings objects and the leaf nodes
    inherit their keyword values from parent nodes up to the root.

    get_closest: Returns either the path or node that is the closest
                 match to a supplied path.

    settings: Returns the inherited resulting Settings from a given
              group.
    """

    def __init__(self, items=None, identifier=None, parent=None, groups=None):
        if groups is None:
            raise ValueError('Please supply groups dictionary')
        self.__dict__['groups'] = groups
        self.__dict__['instantiated'] = False
        AttrTree.__init__(self, items, identifier, parent)
        self.__dict__['instantiated'] = True




    def _process_settings(self, opt_label, label, opts):
        kwargs = {}
        settings = opts.kwargs
        if not self.instantiated:
            settings['valid_keywords'] = opts.valid_keywords
            settings['viewable_name'] = opts.viewable_name
        current_node = self[label] if label in self.children else self
        if current_node.groups.get(opt_label, None) is None:
            if self.instantiated:
                raise Exception("%s does not support %s." % ('.'.join([self.path, label]),
                                                             opt_label))
            else:
                current_node = SettingsTree()
        kwargs[opt_label] = current_node.groups[opt_label](**settings)
        return kwargs


    def __setattr__(self, label, val):
        kwargs = {}
        if isinstance(val, dict):
            for opt_type, opts in val.items():
                kwargs.update(self._process_groups(opt_type, label, opts))
        if kwargs:
            val = SettingsTree(None, identifier=label, parent=self, groups=kwargs)
        else:
            raise ValueError('SettingsTree only accepts a dictionary of Settings.')
        super(SettingsTree, self).__setattr__(label, val)


    def get_closest(self, path, mode='node'):
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


    def settings(self, key):
        if self.groups.get(key, None) is None:
            return None
        if self.parent is None:
            return self.groups[key]
        return Settings(**dict(self.parent.settings(key),
                               **self.groups[key].kwargs))


    def _node_identifier(self, node):
        if node.parent is None:
            return '--+'
        else:
            opts = [node.settings(key) for key, item in node.groups.items()]
            return "%s: " % str(node.identifier) + ', '.join([str(opt) for opt in opts if opts is not None])


    def __repr__(self):
        if len(self) == 0:
            return self._node_identifier(self)
        return super(SettingsTree, self).__repr__()

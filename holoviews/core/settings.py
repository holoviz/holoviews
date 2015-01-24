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

import param
from .tree import AttrTree


class Cycle(param.Parameterized):
    """
    A simple container class that specifies cyclic settings. A typical
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

    def __init__(self, **params):
        super(Cycle, self).__init__(**params)

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



class Settings(param.Parameterized):
    """
    A Settings object holds a collection of keyword options. In
    addition, Settings support (optional) keyword validation as well
    as infinite indexing over the set of supplied cyclic values.

    Settings support inheritance of setting values via the __call__
    method. By calling a Settings object with additional keywords, you
    can create a new Settings object inheriting the parent settings.
    """

    allowed_keywords = param.List(default=None, allow_None=True, doc="""
       Optional list of strings corresponding to the allowed keywords.""")

    key = param.String(default=None, allow_None=True, doc="""
       Optional specification of the settings key name. For
       instance, key could be 'plot' or 'style'.""")


    def __init__(self, allowed_keywords=None, key=None, **kwargs):

        allowed_keywords = sorted(allowed_keywords) if allowed_keywords else None
        super(Settings, self).__init__(allowed_keywords=allowed_keywords, key=key)

        for kwarg in kwargs:
            if allowed_keywords and kwarg not in allowed_keywords:
                raise KeyError("Invalid option %s, valid settings are: %s"
                               % (repr(kwarg), str(self.allowed_keywords)))

        self.kwargs = kwargs
        self._settings = self._expand_settings(kwargs)


    def __call__(self, allowed_keywords=None, **kwargs):
        """
        Create a new Settings object that inherits the parent settings.
        """
        allowed_keywords=self.allowed_keywords if allowed_keywords is None else allowed_keywords
        inherited_style = dict(allowed_keywords=allowed_keywords, **kwargs)
        return self.__class__(key=self.key, **dict(self.kwargs, **inherited_style))


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

    Supports the ability to search the tree for the closest valid path
    using the find method, or compute the appropriate Settings value
    given an object and a mode. For a given node of the tree, the
    settings method computes a Settings object containing the result
    of inheritance for a given group up to the root of the tree.
    """

    def __init__(self, items=None, identifier=None, parent=None, groups=None):
        if groups is None:
            raise ValueError('Please supply groups dictionary')
        self.__dict__['groups'] = groups
        self.__dict__['_instantiated'] = False
        AttrTree.__init__(self, items, identifier, parent)
        self.__dict__['_instantiated'] = True


    def _inherited_settings(self, group_name, settings):
        """
        Computes the inherited Settings object for the given group
        name from the current node given a new set of settings.
        """
        override_kwargs = settings.kwargs
        if not self._instantiated:
            override_kwargs['allowed_keywords'] = settings.allowed_keywords

        if group_name not in self.groups:
            raise KeyError("Group %s not defined on SettingTree" % group_name)

        group_settings = self.groups[group_name]
        try:
            return group_settings(**override_kwargs)
        except KeyError as e:
            keyerror = e.strerror[0].lower() + e.strerror[1:]
            raise KeyError("Invalid key for group %r on path %r; %s )"
                           % (group_name, self.path, keyerror))


    def __setattr__(self, identifier, val):
        new_groups = {}
        if isinstance(val, dict):
            group_items = val
        elif isinstance(val, Settings) and val.key is None:
            raise AttributeError("Settings object needs to have a group name specified.")
        elif isinstance(val, Settings):
            group_items = {val.key: val}

        current_node = self[identifier] if identifier in self.children else self
        for group_name in current_node.groups:
            settings = group_items.get(group_name, False)
            if settings:
                new_groups[group_name] = self._inherited_settings(group_name, settings)
            else:
                new_groups[group_name] = current_node.groups[group_name]

        if new_groups:
            new_node = SettingsTree(None, identifier=identifier, parent=self, groups=new_groups)
        else:
            raise ValueError('SettingsTree only accepts a dictionary of Settings.')
        super(SettingsTree, self).__setattr__(identifier, new_node)


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
        the most appropriate Settings object, including inheritance.

        In addition, closest supports custom settings by checking the
        object
        """
        components = (obj.__class__.__name__, obj.value, obj.label)
        return self.find(components).settings(group)


    def settings(self, group):
        """
        Using inheritance up to the root, get the complete Settings
        object for the given node and the specified group.
        """
        if self.groups.get(group, None) is None:
            return None
        if self.parent is None:
            return self.groups[group]
        return Settings(**dict(self.parent.settings(group),
                               **self.groups[group].kwargs))


    def _node_identifier(self, node):
        if node.parent is None:
            return '--+'
        else:
            values = ', '.join([str(group) for group in node.groups.values()])
            return "%s: %s" % (node.identifier, values)


    def __repr__(self):
        if len(self) == 0:
            return self._node_identifier(self)
        return super(SettingsTree, self).__repr__()

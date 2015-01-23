"""
Options abstract away different classes of options (e.g. matplotlib
specific styles and plot specific parameters) away from View and Map
objects to allow these objects to share options by name.

StyleOpts is an OptionMap that allows matplotlib style options to be
defined, allowing customization of how individual View objects are
displayed given the appropriate style name.
"""

from .tree import AttrTree


class Cycle(object):
    """
    A simple container class to allow specification of cyclic style
    patterns. A typical use would be to cycle styles on a plot with
    multiple curves. Takes either a list of elements or an rckey
    string used to look up the elements in the matplotlib rcParams.g
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
    A Settings object specified keyword options. In addition to the
    functionality of a simple dictionary, Settings support cyclic indexing
    if supplied with a Cycle object.
    """

    def __init__(self, valid_keywords=None, viewable_name=None, **kwargs):
        self.viewable_name = viewable_name
        self.valid_keywords = sorted(valid_keywords) if valid_keywords else None
        if self.valid_keywords is not None:
            for kwarg in kwargs:
                if kwarg not in valid_keywords:
                    err_str_vals = (repr(kwarg), self.viewable_name, str(self.valid_keywords))
                    raise KeyError("Invalid option %s, valid settings for %s are: %s" % err_str_vals)
                
        self.kwargs = kwargs
        self._settings = self._expand_settings(kwargs)


    def __call__(self, valid_keywords=None, viewable_name=None, **kwargs):
        new_style = dict(valid_keywords=self.valid_keywords if valid_keywords is None else valid_keywords,
                         viewable_name=self.viewable_name if viewable_name is None else viewable_name, **kwargs)
        return self.__class__(**dict(self.kwargs, new_style))


    def _expand_settings(self, kwargs):
        """
        Expand out Cycle objects into multiple sets of keyword options.
        """
        filter_static = dict((k,v) for (k,v) in kwargs.items() if not isinstance(v, Cycle))
        filter_cycles = [(k,v) for (k,v) in kwargs.items() if isinstance(v, Cycle)]

        if not filter_cycles: return [kwargs]

        filter_names, filter_values = list(zip(*filter_cycles))
        if not all(len(c)==len(filter_values[0]) for c in filter_values):
            raise Exception("Cycle objects supplied with different lengths")

        cyclic_tuples = list(zip(*[val.elements for val in filter_values]))
        return [dict(zip(filter_names, tps), **filter_static) for tps in cyclic_tuples]


    def keys(self):
        "The keyword names defined in the options."
        return sorted(list(self.kwargs.keys()))


    def __getitem__(self, index):
        """
        Cyclic indexing over any Cycle objects defined.
        """
        return dict(self._settings[index % len(self._settings)])


    @property
    def settings(self):
        if len(self._settings) == 1:
            return dict(self._settings[0])
        else:
            raise Exception("The settings property may only be used with non-cyclic Settings.")


    def __repr__(self):
        kws = ', '.join("%s=%r" % (k,v) for (k,v) in self.kwargs.items())
        return "%s(%s)" % (self.__class__.__name__,  kws)



class SettingsTree(AttrTree):

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

"""
Options abstract away different class of options (e.g. matplotlib
specific styles and plot specific parameters) away from View and Stack
objects, allowing these objects to share options by name.

StyleOpts is an OptionMap that allows matplotlib style options to be
defined, allowing customization of how individual View objects are
displayed if they have the appropriate style name.
"""

from collections import OrderedDict


class Cycle(object):
    """
    A simple container class to allow specification of cyclic style
    patterns. A typical use would be to cycle styles on a plot with
    multiple curves.
    """
    def __init__(self, elements):
        self.elements = elements

    def __len__(self):
        return len(self.elements)

    def __repr__(self):
        return "Cycle(%s)" % self.elements



class Opts(object):
    """
    A Options object specified keyword options. In addition to the
    functionality of a simple dictionary, Opts support cyclic indexing
    if supplied with a Cycle object.
    """

    def __init__(self, **kwargs):
        self.items = kwargs
        self.options = self._expand_styles(kwargs)


    def __call__(self, **kwargs):
        new_style = dict(self.items, **kwargs)
        return self.__class__(**new_style)


    def _expand_styles(self, kwargs):
        """
        Expand out Cycle objects into multiple sets of keyword options.
        """
        filter_static = dict((k,v) for (k,v) in kwargs.items() if not isinstance(v, Cycle))
        filter_cycles = [(k,v) for (k,v) in kwargs.items() if isinstance(v, Cycle)]

        if not filter_cycles: return [kwargs]

        filter_names, filter_values = zip(*filter_cycles)
        if not all(len(c)==len(filter_values[0]) for c in filter_values):
            raise Exception("Cycle objects supplied with different lengths")

        cyclic_tuples = zip(*[val.elements for val in filter_values])
        return [ dict(zip(filter_names, tps), **filter_static) for tps in cyclic_tuples]


    def keys(self):
        "The keyword names defined in the options."
        return self.items.keys()


    def __getitem__(self, index):
        """
        Cyclic indexing over any Cycle objects defined.
        """
        return dict(self.options[index % len(self.options)])


    @property
    def opts(self):
        if len(self.options) == 1:
            return dict(self.options[0])
        else:
            raise Exception("The opts property may only be used with non-cyclic styles")


    def __repr__(self):
        kws = ', '.join("%s=%r" % (k,v) for (k,v) in self.items.items())
        return "%s(%s)" % (self.__class__.__name__,  kws)



class Options(object):
    """
    A Option is a collection of Opts objects that allows convenient
    attribute access and can compose styles through inheritance.
    Options are inherited by finding all matches which end in the
    same substring as the supplied style.

    For example supplying 'Example_View' as a style would match
    these styles (if they are defined):

    'View' : Opts(a=1, b=2)
    'Example_View': Opts(b=3)

    The resulting Opts object inherits a=1 from 'Options' and b=3
    from 'Example_Options'.
    """

    @classmethod
    def normalize_key(self, key):
        """
        Given a key which may contain spaces, such as a view label,
        convert it to a string suitable for attribute access.
        """
        return key.replace(' ', '')


    def __init__(self, name, opt_type):
        if not issubclass(opt_type, Opts):
            raise Exception("The opt_type needs to be a subclass of Opts.")
        self.name = name
        self._settable = False
        self.opt_type = opt_type
        self.__dict__['_items'] = {}


    def __call__(self, obj):

        if isinstance(obj, str):
            name = obj
        elif isinstance(obj.style, list):
            return self.opt_type()
        else:
            name = obj.style

        matches = sorted((len(key), style) for key, style in self._items.items()
                         if name.endswith(key))
        if matches == []:
            return self.opt_type()
        else:
            base_match = matches[0][1]
            for _, match in matches[1:]:
                base_match = base_match(**match.items)
            return base_match


    def options(self):
        """
        The full list of base Style objects in the Options, excluding
        options customized per object.
        """
        return [k for k in self.keys() if not k.startswith('Custom')]


    def __dir__(self):
        """
        Extend dir() to include base options in IPython tab completion.
        """
        default_dir = dir(type(self)) + list(self.__dict__)
        return sorted(set(default_dir + self.options()))


    def __getattr__(self, name):
        """
        Provide attribute access for the Opts in the Options.
        """
        keys = self.__dict__['_items'].keys()
        if name in keys:
            return self[name]
        raise AttributeError(name)


    def __getitem__(self, obj):
        """
        Fuzzy matching allows a more specific key to be matched
        against a general style entry that has a common suffix.
        """
        return self._items[obj]


    def __repr__(self):
        return "<OptionMap containing %d options>" % len(self._items)


    def keys(self):
        """
        The list of all options in the OptionMap, including options
        associated with individual view objects.
        """
        return self._items.keys()


    def values(self):
        """
        All the Style objects in the OptionMap.
        """
        return self._items.values()


    def __contains__(self, k):
        return k in self.keys()


    def set(self, key, value):
        if not self._settable:
            raise Exception("OptionMaps should be set via OptionGroup")
        if not isinstance(value, Opts):
            raise Exception('An OptionMap must only contain Opts.')
        self._items[self.normalize_key(key)] = value



class OptionsGroup(object):
    """
    An OptionsGroup coordinates the setting of OptionMaps to ensure
    they share a common set of keys. While it is safe to access Opts
    from OptionMaps directly, an OptionGroup object must be used to
    set Options when there are multiple different types of Options
    (plot options as distinct from style options for instance).

    When setting Options, it is important to use the appropriate
    subclass of Opts to disambiguate the OptionGroup to be set. For
    instance, PlotOpts will set plotting options while StyleOpts are
    designed for setting style options.
    """


    normalize_key = Options.normalize_key

    def __init__(self, optmaps):

        names = [o.name for o in optmaps]
        if len(set(names)) != len(names):
            raise Exception("OptionMap names must be unique")

        for optmap in optmaps:
            self.__dict__[optmap.name] = optmap

        self.__dict__['_keys'] = set()
        self.__dict__['_opttypes'] = OrderedDict([(optmap.opt_type, optmap) for optmap in optmaps])


    def __setattr__(self, k, v):
        """
        Attribute style addition of Style objects to the OptionMap.
        """
        opttypes = self.__dict__['_opttypes']
        keys = self.__dict__['_keys']
        keys.add(k)

        if type(v) not in opttypes:
            raise Exception("Options of type %s not applicable" % type(v))

        optmap = opttypes[type(v)]
        optmap._settable = True
        optmap.set(k, v)
        optmap._settable = False


    def fuzzy_match_keys(self, name):
        name = Options.normalize_key(name)
        reversed_matches = sorted((len(key), key) for key in self.keys()
                                  if name.endswith(key))[::-1]
        if reversed_matches:
            return zip(*reversed_matches)[1]
        else:
            return []


    def __getattr__(self, k):
        return self[k]


    def __getitem__(self, key):
        if key not in self._keys:
            raise IndexError('Key not available in the OptionGroup')
        retval = tuple(optmap[key] for optmap in self._opttypes.values())
        return retval[0] if len(retval) == 1 else retval


    def __setitem__(self, key, value):
        if type(value) not in self._opttypes:
            raise Exception("Options of type %s not applicable" % type(value))
        optmap = self._opttypes[type(value)]
        optmap._settable = True
        optmap.set(key, value)
        optmap._settable = False
        self._keys.add(key)


    def options(self):
        """
        The full list of option keys in the OptionGroup, excluding
        options customized per object.
        """
        return [k for k in self.keys() if not k.startswith('Custom')]


    def keys(self):
        return sorted(list(self._keys))


    def __dir__(self):
        """
        Extend dir() to include base options in IPython tab completion.
        """
        default_dir = dir(type(self)) + list(self.__dict__)
        names = [o.name for o in self._opttypes.values()]
        return sorted(set(default_dir + self.keys() + names))



class StyleOpts(Opts):
    """
    A subclass of Opts designed to hold matplotlib options to set the
    display Style of View objects.
    """


class PlotOpts(Opts):
    """
    A subclass of Opts designed to hold plotting options that set the
    parameters of the Plot class that display View objects.
    """


class ChannelOpts(Opts):
    """
    A subclass of Opts designed to hold channel mode definitions that
    control how particular labelled layer combinations in an Overlay
    are displayed.
    """
    def __init__(self, mode, pattern, **kwargs):
        self.mode = mode
        self.pattern = pattern
        self.size = len(pattern.rsplit('*'))
        self.options = self._expand_styles(kwargs)
        self._kwargs = kwargs

    def __repr__(self):
        return "%s(%s%s)" % (self.__class__.__name__,
                              self.mode+(', ' if self._kwargs else ''),
                              self.keywords)



channels = OptionsGroup([Options('definitions', ChannelOpts)])
options = OptionsGroup([Options('plotting', PlotOpts),
                        Options('style', StyleOpts)])

# Default Styles
options.Style = StyleOpts()
options.Contours = StyleOpts(color=Cycle(['k', 'w']))
options.SheetView = StyleOpts(cmap='gray', interpolation='nearest')
options.Curve = StyleOpts(color=Cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k']), linewidth=2)
options.Annotation = StyleOpts()
options.Histogram = StyleOpts(ec='k', fc='w')
options.Table = StyleOpts()

# Defining the most common style options for dataviews
GrayNearest = StyleOpts(cmap='gray', interpolation='nearest')

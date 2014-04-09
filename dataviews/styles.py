"""
Styles abstract away matplotlib specific options from Views and
Stacks, allowing Views to share styles by name. New Styles may be
added to allow customization of display for individual view objects by
adding the customized Style object to the StyleMap and linking the
appropriate style name to the object.
"""


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


class Style(object):
    """
    A Style object controls the matplotlib display options for a given
    View object.
    """
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self.styles = self._expand_styles(kwargs)

    def __call__(self, **kwargs):
        new_style = dict(self._kwargs, **kwargs)
        return self.__class__(**new_style)

    def _expand_styles(self, kwargs):

        filter_static = dict((k,v) for (k,v) in kwargs.items() if not isinstance(v, Cycle))
        filter_cycles = [(k,v) for (k,v) in kwargs.items() if isinstance(v, Cycle)]

        if not filter_cycles: return [kwargs]

        filter_names, filter_values = zip(*filter_cycles)
        if not all(len(c)==len(filter_values[0]) for c in filter_values):
            raise Exception("Cycle objects supplied with different lengths")

        cyclic_tuples = zip(*[val.elements for val in filter_values])
        return [ dict(zip(filter_names, tps), **filter_static) for tps in cyclic_tuples]


    def __getitem__(self, index):
        return dict(self.styles[index % len(self.styles)])

    @property
    def opts(self):
        if len(self.styles) == 1:
            return dict(self.styles[0])
        else:
            raise Exception("The opts property may only be used with non-cyclic styles")

    def __repr__(self):
        keywords = ', '.join("%s=%s" % (k,v) for (k,v) in self._kwargs.items())
        return "Style(%s)" % keywords



class StyleMap(object):
    """
    A StyleMap is a collection of Styles that allows convenient
    attribute access. Styles can be accessed by attribute name and new
    Styles can be added to the StyleMap assigning a Style object to a
    a new attribute name.

    The StyleMap also allows indexing for fuzzy matching access. If
    the key matches a known Style exactly, it will be returned as
    would be expected. If there is no exact match but a Style with a
    more specific name (i.e. there is an arbitrary prefix before the
    supplied key) then this Style will be returned instead. This
    simple mechanism allows a hierarchy of style classes of different
    generality.
    """
    def __init__(self):
        self.__dict__['items'] = {}


    def fuzzy_matches(self, name):
        reversed_matches = sorted((len(key), key) for key in self.items.keys()
                                  if name.endswith(key))[::-1]
        if reversed_matches:
            return zip(*reversed_matches)[1]
        else:
            return []


    def fuzzy_match_style(self, name):
        matches = sorted((len(key), style) for key, style in self.items.items()
                         if name.endswith(key))[::-1]

        if matches == []:
            return Style()
        else:
            return matches[0][1]


    def styles(self):
        """
        The full list of base Style objects in the StyleMap, excluding
        styles customized per object.
        """
        return [k for k in self.keys() if not k.startswith('Custom')]


    def __dir__(self):
        """
        Extend dir() to include base styles in IPython tab completion.
        """
        default_dir = dir(type(self)) + list(self.__dict__)
        return sorted(set(default_dir + self.styles()))


    def __getattr__(self, name):
        """
        Provide attribute access for the styles in the StyleMap.
        """
        keys = self.__dict__['items'].keys()
        if name in keys:
            return self[name]
        raise AttributeError(name)


    def __setattr__(self, k, v):
        """
        Attribute style addition of Style objects to the StyleMap.
        """
        items = self.__dict__['items']
        if k in items.keys():
            self[k] =v
        else:
            items[k] = v


    def __setitem__(self, key, value):
        if not isinstance(value, Style):
            raise Exception('A StyleMap must only contain Style objects.')
        self.items[key.replace(' ', '_')] = value


    def __getitem__(self, obj):
        """
        Fuzzy matching allows a more specific key to be matched
        against a general style entry that has a common suffix.
        """
        if isinstance(obj, str):
            return self.fuzzy_match_style(obj)
        else:
            return self.fuzzy_match_style(obj.style)

    def __repr__(self):
        return "<StyleMap containing %d styles>" % len(self.items)


    def keys(self):
        """
        The list of all styles in the StyleMap, including styles
        associated with individual view objects.
        """
        return self.items.keys()


    def values(self):
        """
        All the Style objects in the StyleMap.
        """
        return self.items.values()

    def __contains__(self, k):
        return k in self.keys()


Styles = StyleMap()
Styles['Default'] = Style()

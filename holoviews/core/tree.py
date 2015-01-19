from collections import OrderedDict

from .view import View


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

    def __init__(self, identifier=None, parent=None, items=None):
        """
        identifier: A string identifier for the current node (if any)
        parent:     The parent node (if any)
        items:      Items as (path, value) pairs to construct
                    (sub)tree down to given leaf values.

        Note that the root node does not have a parent and does not
        require an identifier.
        """
        self.__dict__['parent'] = parent
        self.__dict__['identifier'] = self._valid_identifier(identifier)
        self.__dict__['children'] = []
        self.__dict__['_fixed'] = False

        fixed_error = 'No attribute %r in this AttrTree, and none can be added because fixed=True'
        self.__dict__['_fixed_error'] = fixed_error
        self.__dict__['data'] = OrderedDict()
        items = [] if items is None else (items if isinstance(items, list) else items.items())
        for path, item in items:
            self.set_path(path, item)


    def __iter__(self):
        return iter(self.data.values())


    def __contains__(self, name):
        return name in self.children or name in self.data


    def __len__(self):
        return len(self.data)


    def _valid_identifier(self, identifier):
        """
        Replace spaces with underscores and returns value after
        checking validity.
        """
        if identifier is None: return
        identifier = identifier.replace(' ', '_')
        invalid_chars = any(not el.isalnum() and el!='_' for el in identifier)
        if invalid_chars or not identifier[0].isalpha():
            raise SyntaxError("Invalid Python identifier: %r" % identifier)
        return identifier


    @property
    def fixed(self):
        "If fixed, no new paths can be created via attribute access"
        return self.__dict__['_fixed']

    @fixed.setter
    def fixed(self, val):
        self.__dict__['_fixed'] = val


    def update(self, other):
        """
        Updated the contents of the current AttrTree with the
        contents of a second AttrTree.
        """
        fixed_status = (self.fixed, other.fixed)
        (self.fixed, other.fixed) = (False, False)
        if self.parent is None:
            self.data.update(other.data)
        for identifier in other.children:
            item = other[identifier]
            if identifier not in self:
                self[identifier] = item
            else:
                self[identifier].update(item)
        (self.fixed, other.fixed) = fixed_status


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


    def filter(self, path_filters):
        """
        Filters the loaded AttrTree using the supplied path_filters.
        """
        if not path_filters: return self

        # Convert string path filters
        path_filters = [tuple(pf.split('.')) if not isinstance(pf, tuple)
                        else pf for pf in path_filters]

        # Search for substring matches between paths and path filters
        new_attrtree = self.__class__()
        for path, item in self.data.items():
            if any([all([subpath in path for subpath in pf]) for pf in path_filters]):
                new_attrtree.set_path(path, item)

        return new_attrtree


    def _propagate(self, path, val):
        """
        Propagate the value up to the root node.
        """
        self.data[path] = val
        if self.parent is not None:
            self.parent._propagate((self.identifier,)+path, val)


    def __setitem__(self, identifier, val):
        """
        Set a value at a child node with given identifier. If at a root
        node, multi-level path specifications is allowed (i.e. 'A.B.C'
        format or tuple format) in which case the behaviour matches
        that of set_path.
        """
        if isinstance(identifier, str) and '.' not in identifier:
            self.__setattr__(identifier, val)
        elif isinstance(identifier, str) and self.parent is None:
            self.set_path(tuple(identifier.split('.')), val)
        elif isinstance(identifier, tuple) and self.parent is None:
            self.set_path(identifier, val)
        else:
            raise Exception("Multi-level item setting only allowed from root node.")


    def __getitem__(self, identifier):
        """
        For a given non-root node, access a child element by identifier.

        If the node is a root node, you may also access elements using
        either tuple format or the 'A.B.C' string format.
        """
        keyerror_msg = ''
        split_label = (tuple(identifier.split('.'))
                       if isinstance(identifier, str) else tuple(identifier))
        if len(split_label) == 1:
            identifier = split_label[0]
            if identifier in self.children:
                return self.__dict__[identifier]
            else:
                raise KeyError(identifier + ((' : %s' % keyerror_msg) if keyerror_msg else ''))
        path_item = self
        for identifier in split_label:
            path_item = path_item[identifier]
        return path_item


    def get(self, identifier, default=None):
        return self.__dict__.get(identifier, default)


    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def pop(self, identifier, default=None):
        if identifier in self.children:
            item = self[identifier]
            self.__delitem__(identifier)
            return item
        else:
            return default


    def __setattr__(self, identifier, val):
        identifier = self._valid_identifier(identifier)
        # Getattr is skipped for root and first set of children
        shallow = (self.parent is None or self.parent.parent is None)
        if identifier[0].isupper() and self.fixed and shallow:
            raise AttributeError(self._fixed_error % identifier)

        super(AttrTree, self).__setattr__(identifier, val)

        if identifier in self.children: pass
        elif identifier[0].isupper():
            self.children.append(identifier)
            self._propagate((identifier,), val)


    def __getattr__(self, identifier):
        """
        Access a identifier from the AttrTree or generate a new AttrTree
        with the chosen attribute path.
        """
        try:
            return super(AttrTree, self).__getattr__(identifier)
        except AttributeError: pass

        if identifier.startswith('_'):   raise AttributeError(str(identifier))
        elif self.fixed==True:           raise AttributeError(self._fixed_error % identifier)

        if identifier in self.children:
            return self.__dict__[identifier]

        if identifier[0].isupper():
            self.children.append(identifier)
            child_tree = self.__class__(identifier=identifier, parent=self)
            self.__dict__[identifier] = child_tree
            return child_tree
        else:
            raise AttributeError("%s: Custom paths elements must be capitalized." % identifier)


    def _draw_tree(self, node, prefix='', identifier=''):
        """
        Recursive function that builds up an ASCII tree given an
        AttrTree node.
        """
        children = node.children if isinstance(node, AttrTree) else []
        if isinstance(node, AttrTree):
            identifier = '--+' if node.identifier is None else node.identifier
        else:
            identifier = identifier + ' : ' + str(type(node).__name__)

        tree =  prefix[:-3] + '  +--' if prefix else prefix
        tree += identifier + '\n'
        for index, child in enumerate(children):
            child_prefix = prefix + ('   ' if index+1 == len(children) else '  |')
            tree += self._draw_tree(node[child], child_prefix, child)
        return tree

    def __repr__(self):
        """
        The repr of an AttrTree is an ASCII tree showing the structure
        of the tree and the types of leaves.
        """
        if len(self) == 0:
            return "Dangling AttrTree node with no leaf items."
        return "%s of %s items:\n\n%s" % (self.__class__.__name__,
                                          len(self.data),
                                          self._draw_tree(self))

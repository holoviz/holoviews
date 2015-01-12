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

    def __init__(self, label=None, parent=None, path_items=None):
        self.__dict__['parent'] = parent
        self.__dict__['label'] = label
        self.__dict__['children'] = []
        self.__dict__['_fixed'] = False

        fixed_error = 'No attribute %r in this AttrTree, and none can be added because fixed=True'
        self.__dict__['_fixed_error'] = fixed_error
        self.__dict__['path_items'] = OrderedDict()
        if path_items:
            path_items = OrderedDict(path_items)
            for path, item in path_items.items():
                self.set_path(path, item)


    def __iter__(self):
        return iter(self.path_items.values())

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
            self.path_items.update(other.path_items)
        for label in other.children:
            item = other[label]
            if label not in self:
                self[label] = item
            else:
                self[label].update(item)
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
        for path, item in self.path_items.items():
            if any([all([subpath in path for subpath in pf]) for pf in path_filters]):
                new_attrtree.set_path(path, item)

        return new_attrtree


    def _propagate(self, path, val):
        """
        Propagate the value up to the root node.
        """
        self.path_items[path] = val
        if self.parent is not None:
            self.parent._propagate((self.label,)+path, val)


    def __setitem__(self, label, val):
        """
        Set a value at a child node with given label. If at a root
        node, multi-level path specifications is allowed (i.e. 'A.B.C'
        format or tuple format) in which case the behaviour matches
        that of set_path.
        """
        if isinstance(label, str) and '.' not in label:
            self.__setattr__(label, val)
        elif isinstance(label, str) and self.parent is None:
            self.set_path(tuple(label.split('.')), val)
        elif isinstance(label, tuple) and self.parent is None:
            self.set_path(label, val)
        else:
            raise Exception("Multi-level item setting only allowed from root node.")


    def __getitem__(self, label):
        """
        For a given non-root node, access a child element by label.

        If the node is a root node, you may also access elements using
        either tuple format or the 'A.B.C' string format.
        """
        keyerror_msg = ''
        split_label = (tuple(label.split('.'))
                       if isinstance(label, str) else tuple(label))
        if len(split_label) == 1:
            label = split_label[0]
            if label in self.children:
                return self.__dict__[label]
            else:
                raise KeyError(label + ((' : %s' % keyerror_msg) if keyerror_msg else ''))
        path_item = self
        for label in split_label:
            path_item = path_item[label]
        return path_item


    def get(self, label, default=None):
        return self.__dict__.get(label, default)


    def keys(self):
        return self.children[:]


    def pop(self, label, default=None):
        if label in self.children:
            item = self[label]
            self.__delitem__(label)
            return item
        else:
            return default


    def __setattr__(self, label, val):
        # Getattr is skipped for root and first set of children
        shallow = (self.parent is None or self.parent.parent is None)
        if label[0].isupper() and self.fixed and shallow:
            raise AttributeError(self._fixed_error % label)

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
        elif self.fixed==True:      raise AttributeError(self._fixed_error % label)

        if label in self.children:
            return self.__dict__[label]

        if label[0].isupper():
            self.children.append(label)
            child_tree = self.__class__(label=label, parent=self)
            self.__dict__[label] = child_tree
            return child_tree
        else:
            raise AttributeError("%s: Custom paths elements must be capitalized." % label)


    def _draw_tree(self, node, prefix='', label=''):
        """
        Recursive function that builds up an ASCII tree given an
        AttrTree node.
        """
        children = node.children if isinstance(node, AttrTree) else []
        if isinstance(node, AttrTree):
            label = 'root' if node.label is None else node.label
        else:
            label = label + ' : ' + str(type(node).__name__)

        tree =  prefix[:-3] + '  +--' if prefix else prefix
        tree += label + '\n'
        for index, child in enumerate(children):
            child_prefix = prefix + ('   ' if index+1 == len(children) else '  |')
            tree += self._draw_tree(node[child], child_prefix, child)
        return tree

    def __repr__(self):
        """
        The repr of an AttrTree is an ASCII tree showing the structure
        of the tree and the types of leaves.
        """
        if len(self.path_items) == 0:
            return "Dangling AttrTree node with no leaf items."
        return self._draw_tree(self)


    def __contains__(self, name):
        return name in self.children or name in self.path_items

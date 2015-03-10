"""
HoloViews can be used to build highly-nested data-structures
containing large amounts of raw data. As a result, it is difficult to
generate a readable representation that is both informative yet
concise.

As a result, HoloViews does not attempt to build representations that
can be evaluated with eval; such representations would typically be
far too large to be practical. Instead, all HoloViews objects can be
represented as tree structures, showing the types, values and labels
where possible.
"""


class PrettyPrinter(object):
    """
    The PrettyPrinter used to print all HoloView objects via the
    pprint classmethod.
    """

    tab = '   '

    @classmethod
    def pprint(cls, node):
        return  cls.serialize(cls.recurse(node))

    @classmethod
    def serialize(cls, lines):
        accumulator = []
        for level, line in lines:
            accumulator.append((level *cls.tab) + line)
        return "\n".join(accumulator)

    @classmethod
    def shift(cls, lines, shift=0):
        return [(lvl+shift, line) for (lvl, line) in lines]

    @classmethod
    def padding(cls, items):
        return max(len(p) for p in items) if len(items) > 1 else len(items[0])

    @classmethod
    def component_type(cls, node):
        "Return the type.group.label dotted information"
        if node is None: return ''
        components = [':'+str(type(node).__name__)]
        return ".".join(components)

    @classmethod
    def recurse(cls, node, attrpath=None, attrpaths=[], siblings=[], level=0, value_dims=True):
        """
        Recursive function that builds up an ASCII tree given an
        AttrTree node.
        """
        level, lines = cls.node_info(node, attrpath, attrpaths, siblings, level, value_dims)
        attrpaths = ['.'.join(k) for k in node.keys()] if  hasattr(node, 'children') else []
        siblings = [node[child] for child in attrpaths]
        for index, attrpath in enumerate(attrpaths):
            lines += cls.recurse(node[attrpath], attrpath, attrpaths=attrpaths,
                                 siblings=siblings, level=level+1, value_dims=value_dims)
        return lines

    @classmethod
    def node_info(cls, node, attrpath, attrpaths, siblings, level, value_dims):
        """
        Given a node, return relevant information.
        """
        if hasattr(node, 'children'):
            (lvl, lines) = (level, [(level, cls.component_type(node))])
        elif getattr(node, '_deep_indexable', False):
            (lvl, lines) = cls.ndmapping_info(node, siblings, level, value_dims)
        else:
            (lvl, lines) = cls.element_info(node, siblings, level, value_dims)

        # The attribute access path acts as a prefix (if applicable)
        if attrpath is not None:
            padding = cls.padding(attrpaths)
            (fst_lvl, fst_line) = lines[0]
            lines[0] = (fst_lvl, attrpath.ljust(padding) +' ' + fst_line)
        return (lvl, lines)


    @classmethod
    def element_info(cls, node, siblings, level, value_dims):
        """
        Return the information summary for an Element. This consists
        of the dotted name followed by an value dimension names.
        """
        info =  cls.component_type(node)
        if siblings:
            padding = cls.padding([cls.component_type(el) for el in siblings])
            info.ljust(padding)
        if len(node.key_dimensions) >= 1:
            info += cls.tab + '[%s]' % ','.join(d.name for d in node.key_dimensions)
        if value_dims and len(node.value_dimensions) >= 1:
            info += cls.tab + '(%s)' % ','.join(d.name for d in node.value_dimensions)
        return level, [(level, info)]


    @classmethod
    def ndmapping_info(cls, node, siblings, level, value_dims):

        key_dim_info = '[%s]' % ','.join(d.name for d in node.key_dimensions)
        first_line = cls.component_type(node) + cls.tab + key_dim_info
        lines = [(level, first_line)]

        additional_lines = []
        if len(node.data) == 0:
            return level, lines
        # .last has different semantics for GridSpace
        last = node.data.values()[-1]
        if hasattr(last, 'children'):
            element_info, additional_lines = None, cls.recurse(last, level=level)
        # NdOverlays, GridSpace, Ndlayouts
        elif last is not None and getattr(last, '_deep_indexable'):
            element_info = cls.component_type(last)
            level, additional_lines = cls.ndmapping_info(last, [], level, value_dims)
        else:
            _, additional_lines = cls.element_info(last, siblings, level, value_dims)
        lines += cls.shift(additional_lines, 1)
        return level, lines

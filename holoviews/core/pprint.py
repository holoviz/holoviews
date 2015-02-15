"""
Holoviews can be used to build highly-nested data-structures
containing large amounts of raw data. As a result, it is difficult to
generate a readable representation that is both informative yet
concise.

As a result, Holoviews does not attempt to build representations that
can be evaluated with eval; such representations would typically be
far too large to be practical. Instead, all Holoviews objects can be
represented as tree structures, showing the types, values and labels
where possible.
"""

class PrintUtils(object):
    """
    Utilities used to assist the pretty printing process.
    """

    tab = '   '

    @classmethod
    def shift(cls, lines, shift=0):
        return [(lvl+shift, line) for (lvl, line) in lines]

    @classmethod
    def isoverlay(cls, node):
        return hasattr(node, 'children') and hasattr(node, '__mul__')

    @classmethod
    def dotted(cls, node):
        "Return the type.value.label dotted information"
        if node is None: return ''
        components = [str(type(node).__name__), node.value]
        if node.label:
            components.append(node.label)
        return ".".join(components)

    @classmethod
    def leaf_padding(cls, siblings, line):
        """
        Given the collection of sibling leaf names, return the padding
        for the entire group.

        For instance, if you have the tree (Pattern.Gaussian +
        Pattern.Disk) then the padding is the length of 'Gaussian'.
        """
        return max(len(el) for el in siblings) if siblings != [] else len(line)

    @classmethod
    def dotted_padding(cls, node, siblings):
        """
        Compute the appropriate padding based on the longest dotted
        name among the supplied siblings.
        """
        if siblings == []: return len(cls.dotted(node))
        return max([len(cls.dotted(sibling)) for sibling in siblings])

    @classmethod
    def layoutree_annotation(cls, level, lines, node_name, node_names):
        "Annotate the first line with the leaf label as appropriate"
        if node_name:
            tree_leaf = node_name.ljust(cls.leaf_padding(node_names, node_name))
            if lines:
                first_level, first_line, = lines[0]
                lines[0] = (first_level, tree_leaf + ' : ' + first_line)
            else:
                lines = [(level, tree_leaf)]
        return level, lines



class Info(object):
    """
    Methods used to supply the pretty printer with information by
    holoviews object type in a suitable format.
    """

    @classmethod
    def overlay_info(cls, level, node, siblings, value_dims=True):
        overlay_info, siblings = [], node.values()
        for el in siblings:
            _, element_info = cls.element_info(level, el, siblings, value_dims=True)
            element_info = [(lvl,'*--'+line) for (lvl,line) in element_info]
            overlay_info += element_info

        return cls.dotted(node), overlay_info


    @classmethod
    def node_info(cls, level, node, siblings, value_dims=True):
        """
        Given a node, return relevant information.
        """
        if hasattr(node, 'children'):
            return level, []
        elif getattr(node, '_deep_indexable', False):
            return cls.ndmapping_info(level, node, siblings)
        else:
            return cls.element_info(level, node, siblings)

    @classmethod
    def element_info(cls, level, node, siblings, value_dims=True):
        """
        Return the information summary for an Element. This consists
        of the dotted name followed by an value dimension names.
        """
        dotted_padding = cls.dotted_padding(node, siblings)
        info =  cls.dotted(node).ljust(dotted_padding)
        if node is not None and value_dims and len(node.value_dimensions) >= 1:
            info += cls.tab + '(%s)' % ','.join(d.name for d in node.value_dimensions)
        return level, [(level, info)]


    @classmethod
    def ndmapping_info(cls, level, node, siblings, value_dims=True):
        key_dim_info = '[%s]' % ','.join(d.name for d in node.key_dimensions)
        first_line = cls.dotted(node) + cls.tab + key_dim_info
        lines = [(level, first_line)]

        additional_lines = []
        if len(node.data) == 0:
            return level, lines

        if hasattr(node.last, 'children'):  # Must be an Overlay
            overlay_info = cls.overlay_info(level, node.last, siblings, value_dims=True)
            element_info, additional_lines = overlay_info
        # NdOverlays, AxisLayout, Ndlayouts
        elif node.last is not None and getattr(node.last, '_deep_indexable'):
            pass # TODO
        else:
            _, [info] = cls.element_info(level, node.last, siblings, value_dims=True)
            _, element_info = info
        lines += [(level+1, '|_ %s' % element_info)] + cls.shift(additional_lines, 2)
        return level, lines



class PrettyPrinter(Info, PrintUtils):
    """
    The PrettyPrinter used to print all HoloView objects via the
    pprint classmethod.
    """

    @classmethod
    def pprint(cls, node):
        accumulator = ''
        for level, line in cls.recurse(node):
            accumulator += (level *cls.tab) + line + "\n"
        return accumulator


    @classmethod
    def recurse(cls, node, node_name=None, level=0, siblings=[], node_names=[]):
        """
        Recursive function that builds up an ASCII tree given an
        AttrTree node.
        """
        if cls.isoverlay(node):
            header, lines = cls.overlay_info(level, node, siblings, value_dims=True)
            lines = [(level, header)] + lines
            children = []
            siblings = node.values()

        else:
            info = cls.process_node(node, level, node_name, node_names, siblings)
            (level, lines, children, siblings) = info

        for index, child_name in enumerate(children):
            lines += cls.recurse(node[child_name], child_name,
                                 level=level+1, siblings=siblings, node_names=children)
        return lines

    @classmethod
    def process_node(cls, node, level, node_name, node_names, siblings):
        level, lines = cls.node_info(level, node, siblings)
        level, lines = cls.layoutree_annotation(level, lines, node_name, node_names)
        children = node.children if  hasattr(node, 'children') else []
        siblings = [node[child] for child in children]
        return level, lines, children, siblings

"""
HoloViews can be used to build highly-nested data-structures
containing large amounts of raw data. As a result, it is difficult to
generate a readable representation that is both informative yet
concise.

As a result, HoloViews does not attempt to build representations that
can be evaluated with eval; such representations would typically be
far too large to be practical. Instead, all HoloViews objects can be
represented as tree structures, showing how to access and index into
your data.

In addition, there are several different ways of
"""

import re
# IPython not required to import ParamPager
from param.ipython import ParamPager


class InfoPrinter(object):
    """
    Class for printing other information related to an object that is
    of use to the user.
    """
    headings = ['\x1b[1;35m%s\x1b[0m', '\x1b[1;32m%s\x1b[0m']
    ansi_escape = re.compile(r'\x1b[^m]*m')
    ppager = ParamPager()
    store = None

    @classmethod
    def get_parameter_info(cls, obj, ansi=False):
        """
        Get parameter information from the supplied class or object.
        """
        info = cls.ppager._param_docstrings(cls.ppager._get_param_info(obj))
        return self.ansi_escape.sub('', info) if not ansi else info

    @classmethod
    def heading(cls, heading_text, char='=', level=0, ansi=False):
        """
        Turn the supplied heading text into a suitable heading with
        optional underline and color.
        """
        heading_color = cls.headings[level] if ansi else '%s'
        if char is None:
            return heading_color % '%s\n' % heading_text
        else:
            heading_ul = char*len(heading_text)
            return heading_color % '%s\n%s' % (heading_text, heading_ul)


    @classmethod
    def info(cls, obj, category, ansi=False):
        """
        Show information about an object in the given category. ANSI
        color codes may be enabled or disabled.
        """
        categories = ['options', 'indexing', 'object', 'all']

        if category not in categories:
            raise Exception('Valid information categories: %s' % ', '.join(categories))

        plot_class = cls.store.registry[type(obj)]

        obj_info = cls.object_info(obj, ansi=ansi)
        heading = '%s Information' % obj.__class__.__name__
        heading_ul = '='*len(heading)
        prefix = '%s\n%s\n%s\n\n%s\n' % (heading_ul, heading, heading_ul, obj_info)
        if category == 'options':
            info = cls.options_info(obj, plot_class, ansi)
        elif category == 'indexing':
            info = cls.indexing_info(obj, ansi)
        elif category == 'object':
            info = cls.object_params(obj, no_heading=True, ansi=ansi)
        elif category == 'all':
            info = "\n".join([cls.indexing_info(obj, ansi), '',
                              cls.object_params(obj, no_heading=False, ansi=ansi), '',
                              cls.options_info(obj, plot_class, ansi)])
        return prefix + info

    @classmethod
    def indexing_info(cls, obj, ansi=False):
        return '\n'.join(['', cls.heading('Indexing', ansi=ansi), '', repr(obj)])

    @classmethod
    def object_params(cls, obj, no_heading=False,  ansi=False):
        obj_name = obj.__class__.__name__
        element = not getattr(obj, '_deep_indexable', False)
        url = ('https://ioam.github.io/holoviews/Tutorials/Elements.html#{obj}'
               if element else 'https://ioam.github.io/holoviews/Tutorials/Containers.html#{obj}')
        link = url.format(obj=obj_name)

        msg = ("\nMore extensive reference may be accessed using help({obj})"
               + " (or {obj}? in IPython)."
               + '\n\nSee {link} for an example of how to use {obj}.\n')

        param_info = cls.get_parameter_info(obj, ansi=ansi)
        info = [ msg.format(obj=obj_name, link=link),
                 cls.heading('%s Parameters' % obj_name,
                             char='-', level=1, ansi=ansi),
                 '', param_info]
        return '\n'.join(([] if no_heading else [cls.heading('%s Object' % obj_name, ansi=ansi)]) + info)

    @classmethod
    def object_info(cls, obj, ansi=False):
        (count, key_dims, val_dims)  = (1, [], [])
        if hasattr(obj, 'key_dimensions'):
            key_dims = [d.name for d in obj.key_dimensions]
        if hasattr(obj, 'value_dimensions'):
            val_dims = [d.name for d in obj.value_dimensions]
        if hasattr(obj, 'values'):
            count = len(obj.values())

        lines = ['Item count:       %s' % count,
                 'Key dimensions:   %s' % (', '.join(key_dims) if key_dims else 'N/A'),
                 'Value dimensions: %s' % (', '.join(val_dims) if val_dims else 'N/A')]
        return '\n'.join(lines)


    @classmethod
    def options_info(cls, obj, plot_class, ansi=False):
        style_heading = 'Style options:'
        if plot_class.style_opts:
            style_info = "\n(Consult matplotlib's documentation for more information.)"
            style_keywords = '\t%s' % ', '.join(plot_class.style_opts)
            style_msg = '%s\n%s' % (style_keywords, style_info)
        else:
            style_msg = '\t<No style options available>'

        param_info = cls.get_parameter_info(plot_class, ansi=ansi)
        param_heading = '\nPlot options [%s]:' % plot_class.name
        return '\n'.join([ '', cls.heading('Options', ansi=ansi),  '',
                           cls.heading(style_heading, char=None, level=1, ansi=ansi),
                           style_msg,
                           cls.heading(param_heading, char=None, level=1, ansi=ansi),
                           param_info])


class PrettyPrinter(object):
    """
    The PrettyPrinter used to print all HoloView objects via the
    pprint classmethod.
    """

    tab = '   '

    type_formatter= ':{type}'

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
        return cls.type_formatter.format(type=str(type(node).__name__))

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

        # The attribute indexing path acts as a prefix (if applicable)
        if attrpath is not None:
            padding = cls.padding(attrpaths)
            (fst_lvl, fst_line) = lines[0]
            lines[0] = (fst_lvl, '.'+attrpath.ljust(padding) +' ' + fst_line)
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
        last = list(node.data.values())[-1]
        if hasattr(last, 'children'):
            additional_lines = cls.recurse(last, level=level)
        # NdOverlays, GridSpace, Ndlayouts
        elif last is not None and getattr(last, '_deep_indexable'):
            level, additional_lines = cls.ndmapping_info(last, [], level, value_dims)
        else:
            _, additional_lines = cls.element_info(last, siblings, level, value_dims)
        lines += cls.shift(additional_lines, 1)
        return level, lines

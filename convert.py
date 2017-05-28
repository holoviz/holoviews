"""
Prototype demo:

python convert.py Conversion_Example.ipynb > foo.py ; ipython foo.py ; open foo.png
"""
import nbconvert, nbformat
from nbconvert.preprocessors import Preprocessor
import sys
import ast
filename = sys.argv[1]


def wrap_cell_expression(source, prefix, suffix):
    """
    If a cell ends in an expression that could be displaying a HoloViews
    object (as determined using the AST), wrap it with a given prefix
    and suffix string.

    If the cell doesn't end in an expression, return the source unchanged.
    """
    cell_output_types = (ast.IfExp, ast.BoolOp, ast.BinOp, ast.Call,
                         ast.Name, ast.Attribute)
    try:
        node = ast.parse(source)
    except SyntaxError:
        return source
    filtered = source.splitlines()
    if node.body != []:
        last_expr = node.body[-1]
        if not isinstance(last_expr, ast.Expr):
            pass # Not an expression
        elif isinstance(last_expr.value, cell_output_types):
            # CAREFUL WITH UTF8!
            expr_start_line = filtered[last_expr.lineno-1]
            modified = (expr_start_line[:last_expr.col_offset]
                        + prefix + expr_start_line[last_expr.col_offset:])
            filtered[last_expr.lineno-1] = modified
            filtered[-1] = filtered[-1]+',{suffix}'.format(suffix=repr(suffix))
            return '\n'.join(filtered)
    return source


def filter_magic(source, magic, strip=True):
    """
    Given the source of a cell, filter out the given magic and collect
    the lines using the magic into a list.

    If strip is True, the IPython syntax part of the magic (e.g %magic
    or %%magic) is stripped from the returned lines.
    """
    filtered, magic_lines=[],[]
    for line in source.splitlines():
        if line.strip().startswith(magic):
            magic_lines.append(line)
        else:
            filtered.append(line)
    if strip:
        magic_lines = [el.replace(magic,'') for el in magic_lines]
    return '\n'.join(filtered), magic_lines


def strip_magics(source):
    """
    Given the source of a cell, filter out all cell and line magics.
    """
    filtered=[]
    for line in source.splitlines():
        if not line.startswith('%') or line.startswith('%%'):
            filtered.append(line)
    return '\n'.join(filtered)


def replace_line_magic(source, magic, template='{line}'):
    """
    Given a cell's source, replace line magics using a formatting
    template, where {line} is the string that follows the magic.
    """
    filtered = []
    for line in source.splitlines():
        if line.strip().startswith(magic):
            substitution = template.format(line=line.replace(magic, ''))
            filtered.append(substitution)
        else:
            filtered.append(line)
    return '\n'.join(filtered)



class MagicConversion(Preprocessor):

    def preprocess_cell(self, cell, resources, index):
        if cell['cell_type'] == 'code':
            source, opts_lines = filter_magic(cell['source'], '%%opts')
            source = replace_line_magic(source, '%output',
                                        template='hv.util.output({line!r})')
            source = strip_magics(source)
            if opts_lines:
                source = wrap_cell_expression(source,
                                              'hv.util.opts(',
                                              repr(' '.join(opts_lines))+')')
            cell['source'] = source
        return cell, resources

    def __call__(self, nb, resources): # Temporary hack around 'enabled' flag
        return self.preprocess(nb,resources)



with open(filename) as f:
    nb = nbformat.read(f, nbformat.NO_CONVERT)
    exporter = nbconvert.PythonExporter()

    exporter.register_preprocessor(MagicConversion())
    source, meta = exporter.from_notebook_node(nb)
    print(source)

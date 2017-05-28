"""
Prototype demo:

python convert.py Conversion_Example.ipynb > foo.py ; ipython foo.py ; open foo.png
"""
import nbconvert, nbformat
from nbconvert.preprocessors import Preprocessor
import sys
import ast
filename = sys.argv[1]



def comment_out_magics(source):
    """
    Utility used to make sure AST parser does not choke on unrecognized
    magics.
    """
    filtered = []
    for line in source.splitlines():
        if line.strip().startswith('%'):
            filtered.append('# ' +  line)
        else:
            filtered.append(line)
    return '\n'.join(filtered)


def wrap_cell_expression(source, template='{expr}'):
    """
    If a cell ends in an expression that could be displaying a HoloViews
    object (as determined using the AST), wrap it with a given prefix
    and suffix string.

    If the cell doesn't end in an expression, return the source unchanged.
    """
    cell_output_types = (ast.IfExp, ast.BoolOp, ast.BinOp, ast.Call,
                         ast.Name, ast.Attribute)
    try:
        node = ast.parse(comment_out_magics(source))
    except SyntaxError:
        return source
    filtered = source.splitlines()
    if node.body != []:
        last_expr = node.body[-1]
        if not isinstance(last_expr, ast.Expr):
            pass # Not an expression
        elif isinstance(last_expr.value, cell_output_types):
            # CAREFUL WITH UTF8!
            expr_end_slice = filtered[last_expr.lineno-1][:last_expr.col_offset]
            expr_start_slice = filtered[last_expr.lineno-1][last_expr.col_offset:]
            start = '\n'.join(filtered[:last_expr.lineno-1]
                              + ([expr_end_slice] if expr_end_slice else []))
            ending = '\n'.join(([expr_start_slice] if expr_start_slice else [])
                            + filtered[last_expr.lineno:])
            return start + template.format(expr=ending)
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



class OptsMagicProcessor(Preprocessor):

    def preprocess_cell(self, cell, resources, index):
        if cell['cell_type'] == 'code':
            source = replace_line_magic(cell['source'], '%opts',
                                        template='hv.util.opts({line!r})')
            source, opts_lines = filter_magic(source, '%%opts')
            if opts_lines:
                template = 'hv.util.opts({options!r}, {{expr}})'.format(
                    options=' '.join(opts_lines))
                source = wrap_cell_expression(source, template)
            cell['source'] = source
        return cell, resources

    def __call__(self, nb, resources): return self.preprocess(nb,resources)


class OutputMagicProcessor(Preprocessor):

    def preprocess_cell(self, cell, resources, index):
        if cell['cell_type'] == 'code':
            source = replace_line_magic(cell['source'], '%output',
                                        template='hv.util.output({line!r})')
            source, output_lines = filter_magic(source, '%%output')
            if output_lines:
                template = 'hv.util.output({{expr}}, {options!r})'.format(
                    options=output_lines[-1])
                source = wrap_cell_expression(source, template)

            cell['source'] = source
        return cell, resources

    def __call__(self, nb, resources): return self.preprocess(nb,resources)


class StripMagicsProcessor(Preprocessor):

    def preprocess_cell(self, cell, resources, index):
        if cell['cell_type'] == 'code':
            cell['source'] = strip_magics(cell['source'])
        return cell, resources

    def __call__(self, nb, resources): return self.preprocess(nb,resources)


with open(filename) as f:
    nb = nbformat.read(f, nbformat.NO_CONVERT)
    exporter = nbconvert.PythonExporter()
    preprocessors = [OptsMagicProcessor(),
                     OutputMagicProcessor(),
                     StripMagicsProcessor()]
    for preprocessor in preprocessors:
        exporter.register_preprocessor(preprocessor)
    source, meta = exporter.from_notebook_node(nb)
    print(source)

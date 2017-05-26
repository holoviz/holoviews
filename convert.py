"""
Prototype demo:

python convert.py Conversion_Example.ipynb > foo.py ; ipython foo.py ; open foo.png
"""
import nbconvert, nbformat
from nbconvert.preprocessors import Preprocessor
import sys
import ast
filename = sys.argv[1]

cell_output_types = (ast.IfExp, ast.BoolOp, ast.BinOp, ast.Call,
                     ast.Name, ast.Attribute)

class Styler(Preprocessor):

    def preprocess_cell(self, cell, resources, index):
        if cell['cell_type'] == 'code':
            filtered=[]
            queued_opts=[]
            for line in cell['source'].splitlines():
                if line.strip().startswith('%%opts'):
                    queued_opts.append(line)
                elif not line.startswith('%'):
                    filtered.append(line)
            source = '\n'.join(filtered)
            try:
                node = ast.parse(source)
            except SyntaxError:
                return cell, resources
            if node.body != []:
                last_expr = node.body[-1]
                if not isinstance(last_expr, ast.Expr): pass # Not an expression
                elif isinstance(last_expr.value, cell_output_types):
                    # CAREFUL WITH UTF8!
                    expr_start_line = filtered[last_expr.lineno-1]
                    modified = (expr_start_line[:last_expr.col_offset]
                                + 'hv.opts(' + expr_start_line[last_expr.col_offset:])
                    filtered[last_expr.lineno-1] = modified

                    opt_str = ' '.join(el.replace('%%opts','') for el in queued_opts)
                    filtered[-1] = filtered[-1]+',{opt_str})'.format(opt_str=repr(opt_str))
            cell['source'] = '\n'.join(filtered)

        return cell, resources

    def __call__(self, nb, resources): # Temporary hack around 'enabled' flag
        return self.preprocess(nb,resources)


with open(filename) as f:
    nb = nbformat.read(f, nbformat.NO_CONVERT)
    exporter = nbconvert.PythonExporter()

    exporter.register_preprocessor(Styler())
    source, meta = exporter.from_notebook_node(nb)
    print(source)

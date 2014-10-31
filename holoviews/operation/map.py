from collections import OrderedDict

import param

from ..core import ViewMap
from ..core.operation import MapOperation
from ..view import Table, Curve


class table_collate(MapOperation):

    collation_dim = param.String(default="")

    def _process(self, stack):
        collate_dim = self.p.collation_dim
        new_dimensions = [d for d in stack.dimensions if d.name != collate_dim]
        nested_stack = stack.split_dimensions([collate_dim]) if new_dimensions else {(): stack}
        collate_dim = stack.dim_dict[collate_dim]

        table = stack.last
        table_dims = table.dimensions
        if isinstance(stack.last, Table):
            outer_dims = table_dims[-2:]
            new_dimensions += [td for td in table_dims if td not in outer_dims]
            entry_keys = [k[-2:] for k in table.data.keys()]
        else:
            outer_dims = ['Label']
            entry_keys = table.data.keys()

        # Generate a ViewMap for every entry in the table
        stack_fn = lambda: ViewMap(**dict(stack.get_param_values(), dimensions=new_dimensions))
        entries = [(entry, (stack_fn() if new_dimensions else None)) for entry in entry_keys]
        stacks = ViewMap(entries, dimensions=outer_dims)
        for new_key, collate_stack in nested_stack.items():
            curve_data = OrderedDict((k, []) for k in entry_keys)
            # Get the x- and y-values for each entry in the ItemTable
            xvalues = [float(k) for k in collate_stack.keys()]
            for x, table in collate_stack.items():
                for label, value in table.data.items():
                    entry_key = label[-2:] if isinstance(table, Table) else label
                    curve_data[entry_key].append(float(value))

            # Generate curves with correct dimensions
            for label, yvalues in curve_data.items():
                settings = dict(dimensions=[collate_dim])
                if isinstance(table, Table):
                    if not isinstance(label, tuple): label = (label,)
                    if not isinstance(new_key, tuple): new_key = (new_key,)
                    settings.update(value=table.value, label=table.label,
                                    dimensions=[collate_dim])
                    key = new_key + label[0:max(0,len(label)-1)]
                    label = label[-2:]
                else:
                    key = new_key
                    value = table.dim_dict[label]
                    settings.update(value=value, label=table.label,
                                    title='{label} - {value}')
                curve = Curve(zip(xvalues, yvalues), **settings)
                if new_dimensions:
                    stacks[label][key] = curve
                else:
                    stacks[label] = curve

        # If there are multiple table entries, generate grid
        if stacks.ndims in [1, 2]:
            stack = stacks.map(lambda x,k: x.last)
        if isinstance(table, Table):
            grid = stacks.grid(stacks.dimension_labels)
        else:
            grid = stacks.grid(['Label'], layout=True, constant_dims=False)
        return [grid]
from collections import OrderedDict

import param

from ..core import ViewMap, NdMapping
from ..core.operation import MapOperation
from ..view import Table, Curve


class table_collate(MapOperation):

    collation_dim = param.String(default="")

    def _process(self, vmap):
        collate_dim = self.p.collation_dim
        new_dimensions = [d for d in vmap.dimensions if d.name != collate_dim]
        nested_map = vmap.split_dimensions([collate_dim]) if new_dimensions else {(): vmap}
        collate_dim = vmap.dim_dict[collate_dim]

        table = vmap.last
        table_dims = table.dimensions
        if isinstance(vmap.last, Table):
            outer_dims = table_dims[-2:]
            new_dimensions += [td for td in table_dims if td not in outer_dims]
            entry_keys = [k[-2:] for k in table.data.keys()]
        else:
            outer_dims = ['Label']
            entry_keys = table.data.keys()

        # Generate a ViewMap for every entry in the table
        map_fn = lambda: ViewMap(**dict(vmap.get_param_values(), dimensions=new_dimensions))
        entries = [(entry, map_fn() if new_dimensions else None) for entry in entry_keys]
        maps = NdMapping(entries, dimensions=outer_dims)
        for new_key, collate_map in nested_map.items():
            curve_data = OrderedDict((k, []) for k in entry_keys)
            # Get the x- and y-values for each entry in the ItemTable
            xvalues = [float(k) for k in collate_map.keys()]
            for x, table in collate_map.items():
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
                                    title=table.title)
                curve = Curve(zip(xvalues, yvalues), **settings)
                if new_dimensions:
                    maps[label][key] = curve
                else:
                    maps[label] = curve

        # If there are multiple table entries, generate grid
        maps = ViewMap(maps.items(), **dict(maps.get_param_values()))
        if isinstance(table, Table):
            if len(maps) > 1:
                grid = maps.grid(maps.dimension_labels)
            else:
                grid = maps.last
        else:
            grid = maps.grid(['Label'], layout=True, constant_dims=False)
        return [grid]

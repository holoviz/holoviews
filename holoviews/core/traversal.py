"""
Advanced utilities for traversing nesting/hierarchical Dimensioned
objects either to inspect the structure of their declared dimensions
or mutate the matching elements.
"""

from operator import itemgetter

from .dimension import Dimension

def create_ndkey(length, indexes, values):
    key = [None] * length
    for i, v in zip(indexes, values):
        key[i] = v
    return tuple(key)

def uniform(obj):
    """
    Finds all common dimension keys in the object including subsets of
    dimensions. If there are is no common subset of dimensions, None
    is returned.
    """
    dim_groups = obj.traverse(lambda x: tuple(x.key_dimensions),
                              ('HoloMap',))
    if dim_groups:
        return all(set(g1) <= set(g2) or set(g1) >= set(g2)
                   for g1 in dim_groups for g2 in dim_groups)
    return True


def unique_dimkeys(obj, default_dim='Frame'):
    """
    Finds all common dimension keys in the object including subsets of
    dimensions. If there are is no common subset of dimensions, None
    is returned.

    Returns the list of dimensions followed by the list of unique
    keys.
    """
    from .ndmapping import NdMapping
    key_dims = obj.traverse(lambda x: (tuple(x.key_dimensions),
                                       list(x.data.keys())), ('HoloMap',))
    if not key_dims:
        return [Dimension(default_dim)], [(0,)]
    dim_groups, keys = zip(*sorted(key_dims, key=lambda x: -len(x[0])))
    subset = all(set(g1) <= set(g2) or set(g1) >= set(g2)
                 for g1 in dim_groups for g2 in dim_groups)
    # Find unique keys
    if subset:
        all_dims = sorted({dim for dim_group in dim_groups
                           for dim in dim_group},
                               key=lambda x: dim_groups[0].index(x))
    else:
        all_dims = [default_dim]

    ndims = len(all_dims)
    unique_keys = []
    for group, keys in zip(dim_groups, keys):
        dim_idxs = [all_dims.index(dim) for dim in group]
        for key in keys:
            padded_key = create_ndkey(ndims, dim_idxs, key)
            matches = [item for item in unique_keys
                       if padded_key == tuple(k if k is None else i
                                              for i, k in zip(item, padded_key))]
            if not matches:
                unique_keys.append(padded_key)

    sorted_keys = NdMapping({key: None for key in unique_keys},
                            key_dimensions=all_dims).data.keys()
    if subset:
        return all_dims, list(sorted_keys)
    else:
        return all_dims, [(i,) for i in range(len(unique_keys))]


def bijective(keys):
    ndims = len(keys[0])
    if ndims <= 1:
        return True
    for idx in range(ndims):
        getter = itemgetter(*(i for i in range(ndims) if i != idx))
        store = []
        for key in keys:
            subkey = getter(key)
            if subkey in store:
                return False
            store.append(subkey)
    return True

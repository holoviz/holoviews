"""
Advanced utilities for traversing nesting/hierarchical Dimensioned
objects either to inspect the structure of their declared dimensions.
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
    Finds all common dimension keys in the object
    including subsets of dimensions. If there are
    is no common subset of dimensions, None is
    returned.
    """
    dim_groups = obj.traverse(lambda x: tuple(x.key_dimensions),
                              ('HoloMap',))
    if dim_groups:
        return all(set(g1) <= set(g2) or set(g1) >= set(g2)
                   for g1 in dim_groups for g2 in dim_groups)
    return True


def unique_dimkeys(obj):
    """
    Finds all common dimension keys in the object
    including subsets of dimensions. If there are
    is no common subset of dimensions, None is
    returned.
    """
    key_dims = obj.traverse(lambda x: (tuple(x.key_dimensions),
                                       (x.data.keys())), ('HoloMap',))
    if not key_dims:
        return [Dimension('Frame')], [(0,)]
    dim_groups, keys = zip(*sorted(key_dims, lambda d, k: len(d)))
    subset = all(set(g1) <= set(g2) or set(g1) >= set(g2)
               for g1 in dim_groups for g2 in dim_groups)
    # Find unique keys
    all_dims = sorted({dim for dim_group in dim_groups
                       for dim in dim_group},
                      lambda x, k: -dim_groups[0].index(k))
    ndims = len(all_dims)
    unique_keys = []
    for group, keys in key_dims:
        dim_idxs = [all_dims.index(dim) for dim in group]
        for k in keys:
            matches = [item for item in unique_keys
                       if k == (itemgetter(*dim_idxs)(item)
                                if len(dim_idxs) != 1 else (item[dim_idxs[0]]),)]
            if not matches:
                unique_keys.append(create_ndkey(ndims, dim_idxs, k))
    if subset:
        return all_dims, unique_keys
    else:
        return ['Frames'], [i for i in range(len(unique_keys))]


from itertools import combinations

from holoviews.core.options import Store, Options


def option_intersections(backend):
    intersections = []
    options = Store.options(backend)
    for k, opts in sorted(options.items()):
        if len(k) > 1: continue
        valid_options = {k: set(o.allowed_keywords)
                         for k, o in opts.groups.items()}
        for g1, g2 in combinations(Options._option_groups, 2):
            intersection = valid_options[g1] & valid_options[g2]
            if intersection:
                intersections.append((k, intersection))
    return intersections

import numpy as np

def compute_sizes(sizes, size_fn, scaling, base_size, srange):
    """
    Scales point sizes according to a scaling factor,
    base size and size_fn, which will be applied before
    scaling.
    """
    sizes = size_fn(sizes)
    return (base_size*scaling**sizes)


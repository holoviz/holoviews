import numpy as np

def compute_sizes(sizes, option, scaling, base_size, srange):
    """
    Scales point sizes according to a scaling factor,
    base size and normalization option. Valid options
    include 'truncate', 'absolute' and 'normalized'.
    """
    if option == 'truncate':
        sizes = np.ma.array(sizes, mask=sizes<=0)
    elif option == 'absolute':
        sizes = np.abs(sizes)
    elif option == 'normalize':
        sizes = (sizes - srange[0]) / srange[1]-srange[0]
    return (base_size*scaling**sizes)


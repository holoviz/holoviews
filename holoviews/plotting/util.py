import fnmatch
import os

def compute_sizes(sizes, size_fn, scaling, base_size):
    """
    Scales point sizes according to a scaling factor,
    base size and size_fn, which will be applied before
    scaling.
    """
    sizes = size_fn(sizes)
    return (base_size*scaling**sizes)


def find_file(folder, filename):
    """
    Find a file given folder and filename.
    """
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, filename):
            matches.append(os.path.join(root, filename))
    return matches[-1]

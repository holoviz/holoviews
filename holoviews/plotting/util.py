from ..core import HoloMap, CompositeOverlay
from ..core.util import match_spec

def compute_sizes(sizes, size_fn, scaling, base_size):
    """
    Scales point sizes according to a scaling factor,
    base size and size_fn, which will be applied before
    scaling.
    """
    sizes = size_fn(sizes)
    return (base_size*scaling**sizes)


def get_sideplot_ranges(plot, element, main, ranges):
    """
    Utility to find the range for an adjoined
    plot given the plot, the element, the
    Element the plot is adjoined to and the
    dictionary of ranges.
    """

    hist_dim = element.get_dimension(0).name
    range_item = main
    if isinstance(main, HoloMap):
        if issubclass(main.type, CompositeOverlay):
            range_item = [hm for hm in main.split_overlays()[1]
                          if hist_dim in hm.dimensions('all', label=True)][0]
    else:
        range_item = HoloMap({0: main}, kdims=['Frame'])
        ranges = match_spec(range_item.last, ranges)

    if hist_dim in ranges:
        main_range = ranges[hist_dim]
    else:
        framewise = plot.lookup_options(range_item.last, 'norm').options.get('framewise')
        if framewise and range_item.get(key, False):
            main_range = range_item.get(key, False).range(hist_dim)
        else:
            main_range = range_item.range(hist_dim)

    # If .main is an NdOverlay or a HoloMap of Overlays get the correct style
    if isinstance(range_item, HoloMap):
        range_item = range_item.last
    if isinstance(range_item, CompositeOverlay):
        range_item = [ov for ov in range_item
                      if hist_dim in ov.dimensions('value', label=True)][0]
    return range_item, main_range

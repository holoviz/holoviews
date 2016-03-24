from __future__ import unicode_literals

import numpy as np
import param

from ..core import (HoloMap, DynamicMap, CompositeOverlay, Layout,
                    GridSpace, NdLayout, Store)
from ..core.util import (match_spec, is_number, wrap_tuple, basestring,
                         get_overlay_spec, unique_iterator, safe_unicode)


def displayable(obj):
    """
    Predicate that returns whether the object is displayable or not
    (i.e whether the object obeys the nesting hierarchy
    """
    if isinstance(obj, HoloMap):
        return not (obj.type in [Layout, GridSpace, NdLayout])
    if isinstance(obj, (GridSpace, Layout, NdLayout)):
        for el in obj.values():
            if not displayable(el):
                return False
        return True
    return True


class Warning(param.Parameterized): pass
display_warning = Warning(name='Warning')

def collate(obj):
    if isinstance(obj, HoloMap):
        display_warning.warning("Nesting %ss within a HoloMap makes it difficult "
                                "to access your data or control how it appears; "
                                "we recommend calling .collate() on the HoloMap "
                                "in order to follow the recommended nesting "
                                "structure shown in the Composing Data tutorial"
                                "(http://git.io/vtIQh)" % obj.type.__name__)
        return obj.collate()
    elif isinstance(obj, (Layout, NdLayout)):
        try:
            display_warning.warning(
                "Layout contains HoloMaps which are not nested in the "
                "recommended format for accessing your data; calling "
                ".collate() on these objects will resolve any violations "
                "of the recommended nesting presented in the Composing Data "
                "tutorial (http://git.io/vqs03)")
            expanded = []
            for el in obj.values():
                if isinstance(el, HoloMap) and not displayable(el):
                    collated_layout = Layout.from_values(el.collate())
                    expanded.extend(collated_layout.values())
            return Layout(expanded)
        except:
            raise Exception(undisplayable_info(obj))
    else:
        raise Exception(undisplayable_info(obj))


def undisplayable_info(obj, html=False):
    "Generate helpful message regarding an undisplayable object"

    collate = '<tt>collate</tt>' if html else 'collate'
    info = "For more information, please consult the Composing Data tutorial (http://git.io/vtIQh)"
    if isinstance(obj, HoloMap):
        error = "HoloMap of %s objects cannot be displayed." % obj.type.__name__
        remedy = "Please call the %s method to generate a displayable object" % collate
    elif isinstance(obj, Layout):
        error = "Layout containing HoloMaps of Layout or GridSpace objects cannot be displayed."
        remedy = "Please call the %s method on the appropriate elements." % collate
    elif isinstance(obj, GridSpace):
        error = "GridSpace containing HoloMaps of Layouts cannot be displayed."
        remedy = "Please call the %s method on the appropriate elements." % collate

    if not html:
        return '\n'.join([error, remedy, info])
    else:
        return "<center>{msg}</center>".format(msg=('<br>'.join(
            ['<b>%s</b>' % error, remedy, '<i>%s</i>' % info])))


def compute_sizes(sizes, size_fn, scaling_factor, scaling_method, base_size):
    """
    Scales point sizes according to a scaling factor,
    base size and size_fn, which will be applied before
    scaling.
    """
    if scaling_method == 'area':
        pass
    elif scaling_method == 'width':
        scaling_factor = scaling_factor**2
    else:
        raise ValueError(
            'Invalid value for argument "scaling_method": "{}". '
            'Valid values are: "width", "area".'.format(scaling_method))
    sizes = size_fn(sizes)
    return (base_size*scaling_factor*sizes)


def get_sideplot_ranges(plot, element, main, ranges):
    """
    Utility to find the range for an adjoined
    plot given the plot, the element, the
    Element the plot is adjoined to and the
    dictionary of ranges.
    """
    key = plot.current_key
    dims = element.dimensions(label=True)
    dim = dims[1] if dims[1] != 'Frequency' else dims[0]
    range_item = main
    if isinstance(main, HoloMap):
        if issubclass(main.type, CompositeOverlay):
            range_item = [hm for hm in main.split_overlays()[1]
                          if dim in hm.dimensions('all', label=True)][0]
    else:
        range_item = HoloMap({0: main}, kdims=['Frame'])
        ranges = match_spec(range_item.last, ranges)

    if dim in ranges:
        main_range = ranges[dim]
    else:
        framewise = plot.lookup_options(range_item.last, 'norm').options.get('framewise')
        if framewise and range_item.get(key, False):
            main_range = range_item[key].range(dim)
        else:
            main_range = range_item.range(dim)

    # If .main is an NdOverlay or a HoloMap of Overlays get the correct style
    if isinstance(range_item, HoloMap):
        range_item = range_item.last
    if isinstance(range_item, CompositeOverlay):
        range_item = [ov for ov in range_item
                      if dim in ov.dimensions('all', label=True)][0]
    return range_item, main_range, dim


def within_range(range1, range2):
    """Checks whether range1 is within the range specified by range2."""
    return ((range1[0] is None or range2[0] is None or range1[0] >= range2[0]) and
            (range1[1] is None or range2[1] is None or range1[1] <= range2[1]))


def validate_sampled_mode(holomaps, dynmaps):
    composite = HoloMap(enumerate(holomaps), kdims=['testing_kdim'])
    holomap_kdims = set(unique_iterator([kd.name for dm in holomaps for kd in dm.kdims]))
    hmranges = {d: composite.range(d) for d in holomap_kdims}
    if any(not set(d.name for d in dm.kdims) <= holomap_kdims
                        for dm in dynmaps):
        raise Exception('In sampled mode DynamicMap key dimensions must be a '
                        'subset of dimensions of the HoloMap(s) defining the sampling.')
    elif not all(within_range(hmrange, dm.range(d)) for dm in dynmaps
                              for d, hmrange in hmranges.items() if d in dm.kdims):
        raise Exception('HoloMap(s) have keys outside the ranges specified on '
                        'the DynamicMap(s).')


def get_dynamic_mode(composite):
    "Returns the common mode of the dynamic maps in given composite object"
    dynmaps = composite.traverse(lambda x: x, [DynamicMap])
    holomaps = composite.traverse(lambda x: x, ['HoloMap'])
    dynamic_modes = [m.call_mode for m in dynmaps]
    dynamic_sampled = any(m.sampled for m in dynmaps)
    if holomaps:
        validate_sampled_mode(holomaps, dynmaps)
    elif dynamic_sampled and not holomaps:
        raise Exception("DynamicMaps in sampled mode must be displayed alongside "
                        "a HoloMap to define the sampling.")
    if len(set(dynamic_modes)) > 1:
        raise Exception("Cannot display composites of DynamicMap objects "
                        "with different interval modes (i.e open or bounded mode).")
    elif dynamic_modes and not holomaps:
        return 'bounded' if dynamic_modes[0] == 'key' else 'open', dynamic_sampled
    else:
        return None, dynamic_sampled


def initialize_sampled(obj, dimensions, key):
    """
    Initializes any DynamicMaps in sampled mode.
    """
    select = dict(zip([d.name for d in dimensions], key))
    try:
        obj.select([DynamicMap], **select)
    except KeyError:
        pass


def save_frames(obj, filename, fmt=None, backend=None, options=None):
    """
    Utility to export object to files frame by frame, numbered individually.
    Will use default backend and figure format by default.
    """
    backend = Store.current_backend if backend is None else backend
    renderer = Store.renderers[backend]
    fmt = renderer.params('fig').objects[0] if fmt is None else fmt
    plot = renderer.get_plot(obj)
    for i in range(len(plot)):
        plot.update(i)
        renderer.save(plot, '%s_%s' % (filename, i), fmt=fmt, options=options)


def dynamic_update(plot, subplot, key, overlay, items):
    """
    Given a plot, subplot and dynamically generated (Nd)Overlay
    find the closest matching Element for that plot.
    """
    match_spec = get_overlay_spec(overlay,
                                  wrap_tuple(key),
                                  subplot.current_frame)
    specs = [(i, get_overlay_spec(overlay, wrap_tuple(k), el))
             for i, (k, el) in enumerate(items)]
    return closest_match(match_spec, specs)


def closest_match(match, specs, depth=0):
    """
    Recursively iterates over type, group, label and overlay key,
    finding the closest matching spec.
    """
    new_specs = []
    match_lengths = []
    for i, spec in specs:
        if spec[0] == match[0]:
            new_specs.append((i, spec[1:]))
        else:
            if is_number(match[0]) and is_number(spec[0]):
                match_length = -abs(match[0]-spec[0])
            elif all(isinstance(s[0], basestring) for s in [spec, match]):
                match_length = max(i for i in range(len(match[0]))
                                   if match[0].startswith(spec[0][:i]))
            else:
                match_length = 0
            match_lengths.append((i, match_length, spec[0]))

    if len(new_specs) == 1:
        return new_specs[0][0]
    elif new_specs:
        depth = depth+1
        return closest_match(match[1:], new_specs, depth)
    else:
        if depth == 0 or not match_lengths:
            return None
        else:
            return sorted(match_lengths, key=lambda x: -x[1])[0][0]


def map_colors(arr, crange, cmap, hex=True):
    """
    Maps an array of values to RGB hex strings, given
    a color range and colormap.
    """
    if crange:
        cmin, cmax = crange
    else:
        cmin, cmax = np.nanmin(arr), np.nanmax(arr)
    arr = (arr - cmin) / (cmax-cmin)
    arr = np.ma.array(arr, mask=np.logical_not(np.isfinite(arr)))
    arr = cmap(arr)
    if hex:
        arr *= 255
        return ["#{0:02x}{1:02x}{2:02x}".format(*(int(v) for v in c[:-1]))
                for c in arr]
    else:
        return arr


def dim_axis_label(dimensions, separator=', '):
    """
    Returns an axis label for one or more dimensions.
    """
    if not isinstance(dimensions, list): dimensions = [dimensions]
    return separator.join([safe_unicode(d.pprint_label)
                           for d in dimensions])

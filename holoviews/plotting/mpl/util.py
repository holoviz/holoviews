import re
import warnings
from distutils.version import LooseVersion

import numpy as np
import matplotlib
from matplotlib import ticker
from matplotlib.transforms import Bbox, TransformedBbox, Affine2D

mpl_version = LooseVersion(matplotlib.__version__)  # noqa

from ...core.util import basestring, _getargspec
from ...element import Raster, RGB


def wrap_formatter(formatter):
    """
    Wraps formatting function or string in
    appropriate matplotlib formatter type.
    """
    if isinstance(formatter, ticker.Formatter):
        return formatter
    elif callable(formatter):
        args = [arg for arg in _getargspec(formatter).args
                if arg != 'self']
        wrapped = formatter
        if len(args) == 1:
            def wrapped(val, pos=None):
                return formatter(val)
        return ticker.FuncFormatter(wrapped)
    elif isinstance(formatter, basestring):
        if re.findall(r"\{(\w+)\}", formatter):
            return ticker.StrMethodFormatter(formatter)
        else:
            return ticker.FormatStrFormatter(formatter)

def unpack_adjoints(ratios):
    new_ratios = {}
    offset = 0
    for k, (num, ratios) in sorted(ratios.items()):
        unpacked = [[] for _ in range(num)]
        for r in ratios:
            nr = len(r)
            for i in range(num):
                unpacked[i].append(r[i] if i < nr else np.nan)
        for i, r in enumerate(unpacked):
            new_ratios[k+i+offset] = r
        offset += num-1
    return new_ratios

def normalize_ratios(ratios):
    normalized = {}
    for i, v in enumerate(zip(*ratios.values())):
        arr = np.array(v)
        normalized[i] = arr/float(np.nanmax(arr))
    return normalized

def compute_ratios(ratios, normalized=True):
    unpacked = unpack_adjoints(ratios)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        if normalized:
            unpacked = normalize_ratios(unpacked)
        sorted_ratios = sorted(unpacked.items())
        return np.nanmax(np.vstack([v for _, v in sorted_ratios]), axis=0)


def axis_overlap(ax1, ax2):
    """
    Tests whether two axes overlap vertically
    """
    b1, t1 = ax1.get_position().intervaly
    b2, t2 = ax2.get_position().intervaly
    return t1 > b2 and b1 < t2


def resolve_rows(rows):
    """
    Recursively iterate over lists of axes merging
    them by their vertical overlap leaving a list
    of rows.
    """
    merged_rows = []
    for row in rows:
        overlap = False
        for mrow in merged_rows:
            if any(axis_overlap(ax1, ax2) for ax1 in row
                   for ax2 in mrow):
                mrow += row
                overlap = True
                break
        if not overlap:
            merged_rows.append(row)
    if rows == merged_rows:
        return rows
    else:
        return resolve_rows(merged_rows)


def fix_aspect(fig, nrows, ncols, title=None, extra_artists=[],
               vspace=0.2, hspace=0.2):
    """
    Calculate heights and widths of axes and adjust
    the size of the figure to match the aspect.
    """
    fig.canvas.draw()
    w, h = fig.get_size_inches()

    # Compute maximum height and width of each row and columns
    rows = resolve_rows([[ax] for ax in fig.axes])
    rs, cs = len(rows), max([len(r) for r in rows])
    heights = [[] for i in range(cs)]
    widths = [[] for i in range(rs)]
    for r, row in enumerate(rows):
        for c, ax in enumerate(row):
            bbox = ax.get_tightbbox(fig.canvas.get_renderer())
            heights[c].append(bbox.height)
            widths[r].append(bbox.width)
    height = (max([sum(c) for c in heights])) + nrows*vspace*fig.dpi
    width = (max([sum(r) for r in widths])) + ncols*hspace*fig.dpi

    # Compute aspect and set new size (in inches)
    aspect = height/width
    offset = 0
    if title and title.get_text():
        offset = title.get_window_extent().height/fig.dpi
    fig.set_size_inches(w, (w*aspect)+offset)

    # Redraw and adjust title position if defined
    fig.canvas.draw()
    if title and title.get_text():
        extra_artists = [a for a in extra_artists
                         if a is not title]
        bbox = get_tight_bbox(fig, extra_artists)
        top = bbox.intervaly[1]
        if title and title.get_text():
            title.set_y((top/(w*aspect)))


def get_tight_bbox(fig, bbox_extra_artists=[], pad=None):
    """
    Compute a tight bounding box around all the artists in the figure.
    """
    renderer = fig.canvas.get_renderer()
    bbox_inches = fig.get_tightbbox(renderer)
    bbox_artists = bbox_extra_artists[:]
    bbox_artists += fig.get_default_bbox_extra_artists()
    bbox_filtered = []
    for a in bbox_artists:
        bbox = a.get_window_extent(renderer)
        if isinstance(bbox, tuple):
            continue
        if a.get_clip_on():
            clip_box = a.get_clip_box()
            if clip_box is not None:
                bbox = Bbox.intersection(bbox, clip_box)
            clip_path = a.get_clip_path()
            if clip_path is not None and bbox is not None:
                clip_path = clip_path.get_fully_transformed_path()
                bbox = Bbox.intersection(bbox,
                                         clip_path.get_extents())
        if bbox is not None and (bbox.width != 0 or
                                 bbox.height != 0):
            bbox_filtered.append(bbox)
    if bbox_filtered:
        _bbox = Bbox.union(bbox_filtered)
        trans = Affine2D().scale(1.0 / fig.dpi)
        bbox_extra = TransformedBbox(_bbox, trans)
        bbox_inches = Bbox.union([bbox_inches, bbox_extra])
    return bbox_inches.padded(pad) if pad else bbox_inches


def get_raster_array(image):
    """
    Return the array data from any Raster or Image type
    """
    if isinstance(image, RGB):
        rgb = image.rgb
        data = np.dstack([np.flipud(rgb.dimension_values(d, flat=False))
                          for d in rgb.vdims])
    else:
        data = image.dimension_values(2, flat=False)
        if type(image) is Raster:
            data = data.T
        else:
            data = np.flipud(data)
    return data

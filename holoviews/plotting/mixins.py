import numpy as np
import pandas as pd

from ..core import Dataset, Dimension, util
from ..core.util import dtype_kind
from ..element import Bars, Graph
from ..element.util import categorical_aggregate2d
from .util import get_axis_padding


class GeomMixin:
    def get_extents(self, element, ranges, range_type="combined", **kwargs):
        """Use first two key dimensions to set names, and all four
        to set the data range.

        """
        kdims = element.kdims
        # loop over start and end points of segments
        # simultaneously in each dimension
        for kdim0, kdim1 in zip(
            [kdims[i].label for i in range(2)], [kdims[i].label for i in range(2, 4)], strict=None
        ):
            new_range = {}
            for kdim in [kdim0, kdim1]:
                # for good measure, update ranges for both start and end kdim
                for r in ranges[kdim]:
                    if r == "factors":
                        new_range[r] = list(
                            util.unique_iterator(list(ranges[kdim0][r]) + list(ranges[kdim1][r]))
                        )
                    else:
                        # combine (x0, x1) and (y0, y1) in range calculation
                        new_range[r] = util.max_range([ranges[kd][r] for kd in [kdim0, kdim1]])
            ranges[kdim0] = new_range
            ranges[kdim1] = new_range
        return super().get_extents(element, ranges, range_type)


class ChordMixin:
    def get_extents(self, element, ranges, range_type="combined", **kwargs):
        """A Chord plot is always drawn on a unit circle."""
        xdim, ydim = element.nodes.kdims[:2]
        if range_type not in ("combined", "data", "extents"):
            return xdim.range[0], ydim.range[0], xdim.range[1], ydim.range[1]
        no_labels = element.nodes.get_dimension(self.label_index) is None and self.labels is None
        rng = 1.1 if no_labels else 1.4
        x0, x1 = util.max_range([xdim.range, (-rng, rng)])
        y0, y1 = util.max_range([ydim.range, (-rng, rng)])
        return (x0, y0, x1, y1)


class DonutMixin:
    """Shared mixin for Donut plot classes across backends.

    Provides the angle computation that turns raw values into
    wedge start/end angles, plus extent calculation that fixes
    the plot to a unit circle.
    """

    @staticmethod
    def _coerce_donut_values(values):
        """Convert nullable numeric inputs to float, preserving missing values."""
        return np.where(pd.isna(values), np.nan, values).astype(float)

    @staticmethod
    def _filter_donut_data(labels, values):
        """Drop rows with missing labels or missing values.

        Returns
        -------
        labels, values, valid_mask
        """
        labels = np.asarray(labels)
        values = DonutMixin._coerce_donut_values(values)
        valid = ~pd.isna(labels) & ~np.isnan(values)
        return labels[valid], values[valid], valid

    def _resolve_center_text(self, values, element):
        """Return the text to display in the donut center."""
        label = self.center_label
        if label is None:
            return None

        values = self._coerce_donut_values(values)
        total = np.nansum(values)
        vdim = element.vdims[0]
        context = {
            "total": total,
            vdim.name: total,
        }
        if label == "total":
            return str(total)
        return label.format(**context)

    def _compute_label_geometry(self, starts, ends):
        """Compute label positions and text-alignment strings for wedge labels.

        Parameters
        ----------
        starts, ends : array-like
            Wedge start/end angles in radians.

        Returns
        -------
        xs, ys : ndarray
            Label x/y co-ordinates.
        aligns : list[str]
            Per-label ``text_align`` values (``"left"`` or ``"right"``).
        """
        starts = np.asarray(starts)
        ends = np.asarray(ends)
        mid = (starts + ends) / 2
        r = self.outer_radius * self.label_radius
        xs = r * np.cos(mid)
        ys = r * np.sin(mid)
        a = mid % (2 * np.pi)

        if self.label_text_align == "auto":
            # Determine auto-alignment based on position.
            # Outside: left-align on right side, right-align on left.
            # Inside: flip so text "hugs" the wedge boundary.
            right_half = (a < np.pi / 2) | (a > 3 * np.pi / 2)
            if self.label_radius >= 1.0:
                aligns = np.where(right_half, "left", "right")
            else:
                aligns = np.where(right_half, "right", "left")
        else:
            aligns = np.full(len(mid), self.label_text_align)

        return xs, ys, aligns.tolist()

    @staticmethod
    def _compute_donut_data(values):
        """Compute start and end angles from raw values.

        Parameters
        ----------
        values : array-like
            Raw slice values (must be non-negative).

        Returns
        -------
        start_angles, end_angles, fractions
        """
        values = DonutMixin._coerce_donut_values(values)
        if len(values) == 0:
            return np.array([]), np.array([]), np.array([])
        total = np.nansum(values)
        if total == 0:
            fracs = np.zeros_like(values)
        else:
            fracs = np.where(np.isnan(values), 0.0, values) / total
        cumulative = np.cumsum(fracs) * 2 * np.pi
        starts = np.concatenate([[0], cumulative[:-1]])
        ends = cumulative
        return starts, ends, fracs

    def _get_axis_dims(self, element):
        return (element.kdims[0], element.vdims[0])

    def _format_donut_labels(self, element, labels):
        """Format donut labels for display using the key dimension formatter."""
        kdim = self._get_axis_dims(element)[0]
        return [label if isinstance(label, str) else kdim.pprint_value(label) for label in labels]

    def _label_value_for_template(self, label):
        """Return a template-friendly label value.

        Datetime labels are converted to Python datetimes so format
        specs like ``{date:%Y-%m-%d}`` work as expected.
        """
        if isinstance(label, np.datetime64):
            return util.dt64_to_dt(label)
        return label

    def _generate_labels(self, element, labels, values, fractions):
        """Generate wedge labels based on show_labels attribute."""
        show = self.show_labels
        formatted_labels = self._format_donut_labels(element, labels)
        if show is True:
            return formatted_labels
        if show is False or show is None:
            return None

        kdim, vdim = self._get_axis_dims(element)
        labels = [
            show.format(
                **{
                    kdim.name: self._label_value_for_template(label),
                    vdim.name: val,
                    "fraction": frac,
                }
            )
            for label, val, frac in zip(labels, values, fractions, strict=True)
        ]
        return labels

    def _prepare_donut_data(self, element):
        """Extract and compute all backend-agnostic donut data.

        Returns
        -------
        dict with keys:
            labels : raw label values (may be datetime, numeric, etc.)
            values : float array of wedge values
            starts : start angles in radians (with start_angle offset)
            ends   : end angles in radians (with start_angle offset)
            fracs  : fractional share of each wedge
            display_labels : string-formatted labels for display/legend
            valid  : boolean mask used to filter rows from the element
        """
        labels, values, valid = self._filter_donut_data(
            element.dimension_values(0), element.dimension_values(1)
        )
        starts, ends, fracs = self._compute_donut_data(values)
        starts = starts + self.start_angle
        ends = ends + self.start_angle
        display_labels = self._format_donut_labels(element, labels)
        return dict(
            labels=labels,
            values=values,
            starts=starts,
            ends=ends,
            fracs=fracs,
            display_labels=display_labels,
            valid=valid,
        )

    def get_extents(self, element, ranges, range_type="combined", **kwargs):
        """Fixed radial extents for a donut chart."""
        for _d, rs in ranges.items():
            rs.pop("factors", None)

        r = self.outer_radius
        if self.show_labels and self.label_radius > 1.0:
            r = self.outer_radius * self.label_radius

        if range_type == "data":
            return (-r, -r, r, r)

        padding = 0 if self.overlaid else self.padding
        xpad, ypad, _ = get_axis_padding(padding)
        px = r * (1 + xpad)
        py = r * (1 + ypad)
        return (-px, -py, px, py)


class HeatMapMixin:
    def get_extents(self, element, ranges, range_type="combined", **kwargs):
        if range_type in ("data", "combined"):
            agg = element.gridded
            xtype = agg.interface.dtype(agg, 0)
            shape = agg.interface.shape(agg, gridded=True)
            if xtype.kind in "SUO":
                x0, x1 = (0 - 0.5, shape[1] - 0.5)
            else:
                x0, x1 = element.range(0)
            ytype = agg.interface.dtype(agg, 1)
            if ytype.kind in "SUO":
                y0, y1 = (-0.5, shape[0] - 0.5)
            else:
                y0, y1 = element.range(1)
            return (x0, y0, x1, y1)
        else:
            return super().get_extents(element, ranges, range_type)


class SpikesMixin:
    def _get_axis_dims(self, element):
        if "spike_length" in self.lookup_options(element, "plot").options:
            return [element.dimensions()[0], None, None]
        return super()._get_axis_dims(element)

    def get_extents(self, element, ranges, range_type="combined", **kwargs):
        opts = self.lookup_options(element, "plot").options
        if len(element.dimensions()) > 1 and "spike_length" not in opts:
            ydim = element.get_dimension(1)
            s0, s1 = ranges[ydim.label]["soft"]
            s0 = min(s0, 0) if util.isfinite(s0) else 0
            s1 = max(s1, 0) if util.isfinite(s1) else 0
            ranges[ydim.label]["soft"] = (s0, s1)
        proxy_dim = None
        if "spike_length" in opts or len(element.dimensions()) == 1:
            proxy_dim = Dimension("proxy_dim")
            length = opts.get("spike_length", self.spike_length)
            if self.batched:
                bs, ts = [], []
                # Iterate over current NdOverlay and compute extents
                # from position and length plot options
                frame = self.current_frame or self.hmap.last
                for el in frame.values():
                    opts = self.lookup_options(el, "plot").options
                    pos = opts.get("position", self.position)
                    bs.append(pos)
                    ts.append(pos + length)
                proxy_range = (np.nanmin(bs), np.nanmax(ts))
            else:
                proxy_range = (self.position, self.position + length)
            ranges["proxy_dim"] = {
                "data": proxy_range,
                "hard": (np.nan, np.nan),
                "soft": proxy_range,
                "combined": proxy_range,
            }
        return super().get_extents(element, ranges, range_type, ydim=proxy_dim)


class AreaMixin:
    def get_extents(self, element, ranges, range_type="combined", **kwargs):
        vdims = element.vdims[:2]
        vdim = vdims[0].label
        if len(vdims) > 1:
            new_range = {}
            for r in ranges[vdim]:
                if r != "values":
                    new_range[r] = util.max_range([ranges[vd.label][r] for vd in vdims])
            ranges[vdim] = new_range
        else:
            s0, s1 = ranges[vdim]["soft"]
            s0 = min(s0, 0) if util.isfinite(s0) else 0
            s1 = max(s1, 0) if util.isfinite(s1) else 0
            ranges[vdim]["soft"] = (s0, s1)
        return super().get_extents(element, ranges, range_type)


class BarsMixin:
    def _get_axis_dims(self, element):
        if element.ndims > 1 and not (self.stacked or not self.multi_level):
            xdims = element.kdims
        else:
            xdims = element.kdims[0]
        return (xdims, element.vdims[0])

    def get_extents(self, element, ranges, range_type="combined", **kwargs):
        """Make adjustments to plot extents by computing
        stacked bar heights, adjusting the bar baseline
        and forcing the x-axis to be categorical.

        """
        if self.batched:
            overlay = self.current_frame
            element = Bars(
                overlay.table(), kdims=element.kdims + overlay.kdims, vdims=element.vdims
            )
            for kd in overlay.kdims:
                ranges[kd.label]["combined"] = overlay.range(kd)

        vdim = element.vdims[0].label
        s0, s1 = ranges[vdim]["soft"]
        s0 = min(s0, 0) if util.isfinite(s0) else 0
        s1 = max(s1, 0) if util.isfinite(s1) else 0
        ranges[vdim]["soft"] = (s0, s1)
        l, b, r, t = super().get_extents(element, ranges, range_type, ydim=element.vdims[0])
        if range_type not in ("combined", "data"):
            return l, b, r, t

        # Compute stack heights
        xdim = element.kdims[0]
        if self.stacked:
            ds = Dataset(element)
            pos_range = ds.select(**{vdim: (0, None)}).aggregate(xdim, function=np.sum).range(vdim)
            neg_range = ds.select(**{vdim: (None, 0)}).aggregate(xdim, function=np.sum).range(vdim)
            y0, y1 = util.max_range([pos_range, neg_range])
        else:
            y0, y1 = ranges[vdim]["combined"]

        x0, x1 = (l, r) if (util.isnumeric(l) or isinstance(l, util.datetime_types)) else ("", "")
        if range_type == "data":
            return (x0, y0, x1, y1)

        padding = 0 if self.overlaid else self.padding
        _, ypad, _ = get_axis_padding(padding)
        y0, y1 = util.dimension_range(
            y0, y1, ranges[vdim]["hard"], ranges[vdim]["soft"], ypad, self.logy
        )
        y0, y1 = util.dimension_range(y0, y1, self.ylim, (None, None))
        return (x0, y0, x1, y1)

    def _get_coords(self, element, ranges, as_string=True):
        """Get factors for categorical axes."""
        gdim = None
        sdim = None
        if element.ndims == 1:
            pass
        elif not self.stacked:
            gdim = element.get_dimension(1)
        else:
            sdim = element.get_dimension(1)

        xdim = element.dimensions()[0]

        xvals = None
        if xdim.values:
            xvals = xdim.values

        if gdim and not sdim:
            if not xvals and not gdim.values:
                xvals, gvals = categorical_aggregate2d._get_coords(element)
            else:
                if gdim.values:
                    gvals = gdim.values
                elif ranges.get(gdim.label, {}).get("factors") is not None:
                    gvals = ranges[gdim.label]["factors"]
                else:
                    gvals = element.dimension_values(gdim, False)
                gvals = np.asarray(gvals)
                if xvals:
                    pass
                elif ranges.get(xdim.label, {}).get("factors") is not None:
                    xvals = ranges[xdim.label]["factors"]
                else:
                    xvals = element.dimension_values(0, False)
                xvals = np.asarray(xvals)
            c_is_str = dtype_kind(xvals) in "SU" or not as_string
            g_is_str = dtype_kind(gvals) in "SU" or not as_string
            xvals = [x if c_is_str else xdim.pprint_value(x) for x in xvals]
            gvals = [g if g_is_str else gdim.pprint_value(g) for g in gvals]
            return xvals, gvals
        else:
            if xvals:
                pass
            elif ranges.get(xdim.label, {}).get("factors") is not None:
                xvals = ranges[xdim.label]["factors"]
            else:
                xvals = element.dimension_values(0, False)
            xvals = np.asarray(xvals)
            c_is_str = dtype_kind(xvals) in "SU" or not as_string
            xvals = [x if c_is_str else xdim.pprint_value(x) for x in xvals]
            return xvals, None


class MultiDistributionMixin:
    def _get_axis_dims(self, element):
        return element.kdims, element.vdims[0]

    def get_extents(self, element, ranges, range_type="combined", **kwargs):
        return super().get_extents(
            element, ranges, range_type, "categorical", ydim=element.vdims[0]
        )


class GraphMixin:
    def _get_axis_dims(self, element):
        if isinstance(element, Graph):
            element = element.nodes
        return element.dimensions()[:2]

    def get_extents(self, element, ranges, range_type="combined", **kwargs):
        return super().get_extents(element.nodes, ranges, range_type)


class WaterfallMixin:
    """Shared mixin for Waterfall plot classes across backends.

    Provides the cumulative-sum computation that turns incremental
    deltas into floating-bar coordinates, plus extent calculation
    that uses the cumulative range (not the raw delta range).
    """

    @staticmethod
    def _compute_waterfall_data(labels, values, show_total, total_label):
        """Compute bottom/top/kind arrays from incremental deltas.

        Returns
        -------
        labels, values, bottoms, tops, kinds, cumulative
        """
        # Build an explicit boolean sentinel mask rather than relying on
        # np.isnan(values), which would misclassify genuine NaN user data.
        n = len(labels)
        is_total = np.zeros(n + (1 if show_total and n > 0 else 0), dtype=bool)
        if show_total and n > 0:
            labels = [*labels, total_label]
            values = np.append(values, 0.0)  # placeholder; overwritten below
            is_total[-1] = True

        safe_values = np.where(is_total, 0.0, np.where(np.isnan(values), 0.0, values))
        cumulative = np.cumsum(safe_values)

        prev = np.concatenate([[0], cumulative[:-1]])
        is_pos = (~is_total) & (safe_values >= 0)
        is_neg = (~is_total) & (safe_values < 0)

        kinds = np.where(is_total, "total", np.where(is_pos, "positive", "negative"))
        if len(kinds) > 0:
            kinds[0] = "start"

        bottoms = np.where(is_neg, cumulative, prev)
        tops = np.where(is_neg, prev, cumulative)

        if len(is_total) > 0 and is_total[-1]:
            bottoms[-1] = np.minimum(0.0, cumulative[-1])
            tops[-1] = np.maximum(0.0, cumulative[-1])

        return labels, values, bottoms, tops, kinds, cumulative

    def _map_colors(self, kinds, values):
        """Return a list of colors for each bar, resolving nullable start_color and total_color."""
        if self.start_color is not None:
            start = self.start_color
        elif len(values) > 0:
            start = self.positive_color if values[0] >= 0 else self.negative_color
        else:
            start = self.positive_color

        total = self.total_color if self.total_color is not None else start

        color_map = {
            "start": start,
            "positive": self.positive_color,
            "negative": self.negative_color,
            "total": total,
        }
        return [color_map[k] for k in kinds]

    def _resolve_total_label(self, element):
        """Return total_label, raising if it collides with an existing category."""
        xdim = element.kdims[0]
        existing = [
            lbl if isinstance(lbl, str) else xdim.pprint_value(lbl)
            for lbl in element.dimension_values(0, expanded=False)
        ]
        if self.total_label in existing:
            raise ValueError(
                f"The total label {self.total_label!r} conflicts with an existing category in "
                f"kdim {xdim.name!r}. Update the total label with `.opts(total_label=...)`"
            )
        return self.total_label

    def _get_axis_dims(self, element):
        return (element.kdims[0], element.vdims[0])

    def get_extents(self, element, ranges, range_type="combined", **kwargs):
        """Y-range derived from cumulative totals, not raw deltas."""
        if range_type not in ("combined", "data") or not len(element):
            return super().get_extents(element, ranges, range_type, **kwargs)

        values = element.dimension_values(1)
        cumsum = np.cumsum(values)
        all_points = np.concatenate([[0], cumsum])
        y0 = np.nanmin(all_points)
        y1 = np.nanmax(all_points)
        x0, x1 = "", ""

        if range_type == "data":
            return (x0, y0, x1, y1)

        vdim = element.vdims[0]
        if vdim.label in ranges:
            padding = 0 if self.overlaid else self.padding
            _, ypad, _ = get_axis_padding(padding)
            y0, y1 = util.dimension_range(
                y0,
                y1,
                ranges[vdim.label].get("hard", (np.nan, np.nan)),
                ranges[vdim.label].get("soft", (np.nan, np.nan)),
                ypad,
                self.logy,
            )
        return (x0, y0, x1, y1)

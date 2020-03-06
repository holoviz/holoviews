from collections import namedtuple

import numpy as np
import param

from param.parameterized import bothmethod

from .core.dimension import OrderedDict
from .core.element import Element, Layout
from .core.options import CallbackError, Store
from .core.overlay import NdOverlay, Overlay
from .core.spaces import GridSpace
from .streams import SelectionExpr, PlotReset, Stream
from .operation.element import function
from .util import DynamicMap
from .util.transform import dim


class _Cmap(Stream):
    cmap = param.Parameter(default=None, allow_None=True)


_Exprs = Stream.define('Exprs', exprs=[])
_Styles = Stream.define('Styles', colors=[], alpha=1.)
_RegionElement = Stream.define("RegionElement", region_element=None)


_SelectionStreams = namedtuple(
    'SelectionStreams',
    'style_stream exprs_stream cmap_streams '
)

class _base_link_selections(param.ParameterizedFunction):
    """
    Baseclass for linked selection functions.

    Subclasses override the _build_selection_streams class method to construct
    a _SelectionStreams namedtuple instance that includes the required streams
    for implementing linked selections.

    Subclasses also override the _expr_stream_updated method. This allows
    subclasses to control whether new selections override prior selections or
    whether they are combined with prior selections
    """

    @bothmethod
    def instance(self_or_cls, **params):
        inst = super(_base_link_selections, self_or_cls).instance(**params)

        # Init private properties
        inst._selection_expr_streams = []
        inst._reset_streams = []

        # Init selection streams
        inst._selection_streams = self_or_cls._build_selection_streams(inst)

        # Init dict of region streams
        inst._region_streams = {}

        return inst

    def _register(self, hvobj):
        """
        Register an Element of DynamicMap that may be capable of generating
        selection expressions in response to user interaction events
        """
        # Create stream that produces element that displays region of selection
        if getattr(hvobj, "_selection_streams", ()):
            self._region_streams[hvobj] = _RegionElement()

        # Create SelectionExpr stream
        expr_stream = SelectionExpr(source=hvobj, index_cols=self.index_cols)
        expr_stream.add_subscriber(
            lambda **kwargs: self._expr_stream_updated(hvobj, **kwargs)
        )
        self._selection_expr_streams.append(expr_stream)

        # Create PlotReset stream
        reset_stream = PlotReset(source=hvobj)
        reset_stream.add_subscriber(
            lambda **kwargs: setattr(self, 'selection_expr', None)
        )
        self._reset_streams.append(reset_stream)

    def __call__(self, hvobj, **kwargs):
        # Apply kwargs as params
        self.param.set_param(**kwargs)

        # Perform transform
        hvobj_selection = self._selection_transform(hvobj.clone(link=False))

        return hvobj_selection

    def _selection_transform(self, hvobj, operations=()):
        """
        Transform an input HoloViews object into a dynamic object with linked
        selections enabled.
        """
        from .plotting.util import initialize_dynamic
        if isinstance(hvobj, DynamicMap):
            callback = hvobj.callback
            ninputs = len(callback.inputs)
            if ninputs == 1:
                child_hvobj = callback.inputs[0]
                if callback.operation:
                    next_op = {'op': callback.operation, 'kwargs': callback.operation_kwargs}
                else:
                    fn = function.instance(fn=callback.callable)
                    next_op = {'op': fn, 'kwargs': callback.operation_kwargs}
                new_operations = (next_op,) + operations
                return self._selection_transform(child_hvobj, new_operations)
            elif ninputs == 2:
                return Overlay([self._selection_transform(el)
                                for el in hvobj.callback.inputs]).collate()

            initialize_dynamic(hvobj)
            if issubclass(hvobj.type, Element):
                self._register(hvobj)
                chart = Store.registry[Store.current_backend][hvobj.type]
                return chart.selection_display(hvobj).build_selection(
                    self._selection_streams, hvobj, operations,
                    self._region_streams.get(hvobj, None),
                )
            else:
                # This is a DynamicMap that we don't know how to recurse into.
                return hvobj
        elif isinstance(hvobj, Element):
            # Register hvobj to receive selection expression callbacks
            chart = Store.registry[Store.current_backend][type(hvobj)]
            if getattr(chart, 'selection_display', None):
                element = hvobj.clone(link=False)
                self._register(element)
                return chart.selection_display(element).build_selection(
                    self._selection_streams, element, operations,
                    self._region_streams.get(element, None),
                )
            return hvobj
        elif isinstance(hvobj, (Layout, Overlay, NdOverlay, GridSpace)):
            data = OrderedDict([(k, self._selection_transform(v, operations))
                                 for k, v in hvobj.items()])
            new_hvobj = hvobj.clone(data)
            if hasattr(new_hvobj, 'collate'):
                new_hvobj = new_hvobj.collate()
            return new_hvobj
        else:
             # Unsupported object
            return hvobj

    @classmethod
    def _build_selection_streams(cls, inst):
        """
        Subclasses should override this method to return a _SelectionStreams
        instance
        """
        raise NotImplementedError()

    def _expr_stream_updated(self, hvobj, selection_expr, bbox, region_element):
        """
        Called when one of the registered HoloViews objects produces a new
        selection expression.  Subclasses should override this method, and
        they should use the input expression to update the `exprs_stream`
        property of the _SelectionStreams instance that was produced by
        the _build_selection_streams.

        Subclasses have the flexibility to control whether the new selection
        express overrides previous selections, or whether it is combined with
        previous selections.
        """
        raise NotImplementedError()


class link_selections(_base_link_selections):
    """
    Operation which automatically links selections between elements
    in the supplied HoloViews object. Can be used a single time or
    be used as an instance to apply the linked selections across
    multiple objects.
    """

    cross_filter_mode = param.Selector(
        ['overwrite', 'intersect'], default='intersect', doc="""
        Determines how to combine selections across different
        elements.""")

    index_cols = param.List(default=None, doc="""
        If provided, selection switches to index mode where all queries
        are expressed solely in terms of discrete values along the
        index_cols.  All Elements given to link_selections must define the index_cols, either as explicit dimensions or by sharing an underlying Dataset that defines them.""")

    selection_expr = param.Parameter(default=None, doc="""
        dim expression of the current selection or None to indicate
        that everything is selected.""")

    selected_color = param.Color(default=None, allow_None=True, doc="""
        Color of selected data, or None to use the original color of
        each element.""")

    selection_mode = param.Selector(
        ['overwrite', 'intersect', 'union', 'inverse'], default='overwrite', doc="""
        Determines how to combine successive selections on the same
        element.""")

    show_regions = param.Boolean(default=True, doc="""
        Whether to highlight the selected regions.""")

    unselected_alpha = param.Magnitude(default=0.1, doc="""
        Alpha of unselected data.""")

    unselected_color = param.Color(default=None, doc="""
        Color of unselected data.""")

    @bothmethod
    def instance(self_or_cls, **params):
        inst = super(link_selections, self_or_cls).instance(**params)

        # Initialize private properties
        inst._obj_selections = {}
        inst._obj_regions = {}
        inst._reset_regions = True

        return inst

    @classmethod
    def _build_selection_streams(cls, inst):
        # Colors stream
        style_stream = _Styles(
            colors=[inst.unselected_color, inst.selected_color],
            alpha=inst.unselected_alpha
        )

        # Cmap streams
        cmap_streams = [
            _Cmap(cmap=inst.unselected_cmap),
            _Cmap(cmap=inst.selected_cmap),
        ]

        def update_colors(*_):
            colors = [inst.unselected_color, inst.selected_color]
            style_stream.event(colors=colors, alpha=inst.unselected_alpha)
            cmap_streams[0].event(cmap=inst.unselected_cmap)
            if cmap_streams[1] is not None:
                cmap_streams[1].event(cmap=inst.selected_cmap)

        inst.param.watch(update_colors,['unselected_color', 'selected_color', 'unselected_alpha'])

        # Exprs stream
        exprs_stream = _Exprs(exprs=[True, None])

        def update_exprs(*_):
            exprs_stream.event(exprs=[True, inst.selection_expr])
            # Reset regions
            if inst._reset_regions:
                for k, v in inst._region_streams.items():
                    inst._region_streams[k].event(region_element=None)
                inst._obj_selections.clear()
                inst._obj_regions.clear()

        inst.param.watch(update_exprs, ['selection_expr'])

        return _SelectionStreams(
            style_stream=style_stream,
            exprs_stream=exprs_stream,
            cmap_streams=cmap_streams,
        )

    @property
    def unselected_cmap(self):
        """
        The datashader colormap for unselected data
        """
        if self.unselected_color is None:
            return None
        return _color_to_cmap(self.unselected_color)

    @property
    def selected_cmap(self):
        """
        The datashader colormap for selected data
        """
        return None if self.selected_color is None else _color_to_cmap(self.selected_color)

    def _expr_stream_updated(self, hvobj, selection_expr, bbox, region_element):
        if selection_expr:
            if self.cross_filter_mode == "overwrite":
                # clear other regions and selections
                for k, v in self._region_streams.items():
                    if k is not hvobj:
                        self._region_streams[k].event(region_element=None)
                        self._obj_regions.pop(k, None)
                        self._obj_selections.pop(k, None)

            # Update selection expression
            if hvobj not in self._obj_selections or self.selection_mode == "overwrite":
                if self.selection_mode == "inverse":
                    self._obj_selections[hvobj] = ~selection_expr
                else:
                    self._obj_selections[hvobj] = selection_expr
            else:
                if self.selection_mode == "intersect":
                    self._obj_selections[hvobj] &= selection_expr
                elif self.selection_mode == "union":
                    self._obj_selections[hvobj] |= selection_expr
                else:  # inverse
                    self._obj_selections[hvobj] &= ~selection_expr

            # Update region
            if self.show_regions:
                if isinstance(hvobj, DynamicMap):
                    el_type = hvobj.type
                else:
                    el_type = hvobj

                region_element = el_type._merge_regions(
                    self._obj_regions.get(hvobj, None), region_element, self.selection_mode
                )
                self._obj_regions[hvobj] = region_element
            else:
                region_element = None

            # build combined selection
            selection_exprs = list(self._obj_selections.values())
            if self.index_cols:
                if len(selection_exprs) > 1:
                    vals = set.intersection(*(set(expr.ops[2]['args'][0]) for expr in selection_exprs))
                    old = selection_exprs[0]
                    selection_expr = dim('new')
                    selection_expr.dimension = old.dimension
                    selection_expr.ops = list(old.ops)
                    selection_expr.ops[2] = dict(selection_expr.ops[2], args=(list(vals),))
            else:
                selection_expr = selection_exprs[0]
                for expr in selection_exprs[1:]:
                    selection_expr = selection_expr & expr

            # Set _reset_regions to False so that plot regions aren't automatically
            # cleared when self.selection_expr is set.
            self._reset_regions = False
            self.selection_expr = selection_expr
            self._reset_regions = True

            # update this region stream
            if self._region_streams.get(hvobj, None) is not None:
                self._region_streams[hvobj].event(region_element=region_element)


class SelectionDisplay(object):
    """
    Base class for selection display classes.  Selection display classes are
    responsible for transforming an element (or DynamicMap that produces an
    element) into a HoloViews object that represents the current selection
    state.
    """

    def __call__(self, element):
        return self

    def build_selection(self, selection_streams, hvobj, operations, region_stream=None):
        raise NotImplementedError()


class NoOpSelectionDisplay(SelectionDisplay):
    """
    Selection display class that returns input element unchanged. For use with
    elements that don't support displaying selections.
    """

    def build_selection(self, selection_streams, hvobj, operations, region_stream=None):
        return hvobj


class OverlaySelectionDisplay(SelectionDisplay):
    """
    Selection display base class that represents selections by overlaying
    colored subsets on top of the original element in an Overlay container.
    """

    def __init__(self, color_prop='color', is_cmap=False, supports_region=True):
        if not isinstance(color_prop, (list, tuple)):
            self.color_props = [color_prop]
        else:
            self.color_props = color_prop
        self.is_cmap = is_cmap
        self.supports_region = supports_region

    def _get_color_kwarg(self, color):
        return {color_prop: [color] if self.is_cmap else color
                for color_prop in self.color_props}

    def build_selection(self, selection_streams, hvobj, operations, region_stream=None):
        from .element import Histogram

        num_layers = len(selection_streams.style_stream.colors)
        if not num_layers:
            return Overlay()

        layers = []
        for layer_number in range(num_layers):
            streams = [selection_streams.exprs_stream]
            obj = hvobj.clone(link=False) if layer_number == 1 else hvobj
            layer = obj.apply(
                self._build_layer_callback, streams=streams,
                layer_number=layer_number, per_element=True
            )
            layers.append(layer)

        # Wrap in operations
        for op in operations:
            op, kws = op['op'], op['kwargs']
            for layer_number in range(num_layers):
                streams = list(op.streams)
                cmap_stream = selection_streams.cmap_streams[layer_number]
                kwargs = dict(kws)

                # Handle cmap as an operation parameter
                if 'cmap' in op.param or 'cmap' in kwargs:
                    if layer_number == 0 or (op.cmap is None and kwargs.get('cmap') is None):
                        streams += [cmap_stream]
                    else:
                        @param.depends(cmap=cmap_stream.param.cmap)
                        def update_cmap(cmap, default=op.cmap, kw=kwargs.get('cmap')):
                            return cmap or kw or default
                        kwargs['cmap'] = update_cmap
                new_op = op.instance(streams=streams)
                layers[layer_number] = new_op(layers[layer_number], **kwargs)

        for layer_number in range(num_layers):
            layer = layers[layer_number]
            cmap_stream = selection_streams.cmap_streams[layer_number]
            streams = [selection_streams.style_stream, cmap_stream]
            layer = layer.apply(
                self._apply_style_callback, layer_number=layer_number,
                streams=streams, per_element=True
            )
            layers[layer_number] = layer

        # Build region layer
        if region_stream is not None and self.supports_region:
            def update_region(element, region_element, colors, **kwargs):
                unselected_color = colors[0]
                if region_element is None:
                    region_element = element._get_selection_expr_for_stream_value()[2]
                return self._style_region_element(region_element, unselected_color)

            streams = [region_stream, selection_streams.style_stream]
            region = hvobj.clone(link=False).apply(update_region, streams)
            if getattr(hvobj, '_selection_dims', None) == 1 or isinstance(hvobj, Histogram):
                layers.insert(1, region)
            else:
                layers.append(region)
        return Overlay(layers).collate()

    def _build_layer_callback(self, element, exprs, layer_number, **kwargs):
        return self._select(element, exprs[layer_number])

    def _apply_style_callback(self, element, layer_number, colors, cmap, alpha, **kwargs):
        opts = {}
        if layer_number == 0:
            opts['colorbar'] = False
        else:
            alpha = 1
        if cmap is not None:
            opts['cmap'] = cmap
        color = colors[layer_number] if colors else None
        return self._build_element_layer(element, color, alpha, **opts)

    def _build_element_layer(self, element, layer_color, layer_alpha, selection_expr=True):
        raise NotImplementedError()

    def _style_region_element(self, region_element, unselected_cmap):
        raise NotImplementedError()

    @staticmethod
    def _select(element, selection_expr):
        from .element import Curve, Spread
        from .util.transform import dim
        if isinstance(selection_expr, dim):
            dataset = element.dataset
            try:
                if dataset.interface.gridded:
                    mask = selection_expr.apply(dataset, expanded=True, flat=False, strict=True)
                    selection = dataset.clone(dataset.interface.mask(dataset, ~mask))
                elif isinstance(element, (Curve, Spread)) and hasattr(dataset.interface, 'mask'):
                    mask = selection_expr.apply(dataset, compute=False, strict=True)
                    selection = dataset.clone(dataset.interface.mask(dataset, ~mask))
                else:
                    mask = selection_expr.apply(dataset, compute=False, keep_index=True, strict=True)
                    selection = dataset.select(selection_mask=mask)
                element = element.pipeline(selection)
            except KeyError as e:
                key_error = str(e).replace('"', '').replace('.', '')
                raise CallbackError("linked_selection aborted because it could not "
                                    "display selection for all elements: %s on '%r'."
                                    % (key_error, element))
            except Exception as e:
                raise CallbackError("linked_selection aborted because it could not "
                                    "display selection for all elements: %s." % e)
        return element


class ColorListSelectionDisplay(SelectionDisplay):
    """
    Selection display class for elements that support coloring by a
    vectorized color list.
    """

    def __init__(self, color_prop='color', alpha_prop='alpha', backend=None):
        self.color_props = [color_prop]
        self.alpha_props = [alpha_prop]
        self.backend = backend

    def build_selection(self, selection_streams, hvobj, operations, region_stream=None):
        def _build_selection(el, colors, alpha, exprs, **kwargs):
            from .plotting.util import linear_gradient
            ds = el.dataset
            selection_exprs = exprs[1:]
            unselected_color = colors[0]

            # Use darker version of unselected_color if not selected color provided
            unselected_color = unselected_color or "#e6e9ec"
            backup_clr = linear_gradient(unselected_color, "#000000", 7)[2]
            selected_colors = [c or backup_clr for c in colors[1:]]
            n = len(ds)
            clrs = np.array([unselected_color] + list(selected_colors))

            color_inds = np.zeros(n, dtype='int8')

            for i, expr, color in zip(range(1, len(clrs)), selection_exprs, selected_colors):
                if not expr:
                    color_inds[:] = i
                else:
                    color_inds[expr.apply(ds)] = i

            colors = clrs[color_inds]
            color_opts = {color_prop: colors for color_prop in self.color_props}
            return el.pipeline(ds).opts(backend=self.backend, clone=True, **color_opts)

        sel_streams = [selection_streams.style_stream, selection_streams.exprs_stream]
        hvobj = hvobj.apply(_build_selection, streams=sel_streams, per_element=True)

        for op in operations:
            hvobj = op(hvobj)

        return hvobj


def _color_to_cmap(color):
    """
    Create a light to dark cmap list from a base color
    """
    from .plotting.util import linear_gradient
    # Lighten start color by interpolating toward white
    start_color = linear_gradient("#ffffff", color, 7)[2]

    # Darken end color by interpolating toward black
    end_color = linear_gradient(color, "#000000", 7)[2]
    return linear_gradient(start_color, end_color, 64)

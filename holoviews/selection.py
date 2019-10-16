from collections import namedtuple
import copy
import numpy as np
import param

from param.parameterized import bothmethod

from .core import OperationCallable, Overlay
from .core.element import Element, Layout
from .core.options import Store
from .streams import SelectionExpr, Stream
from .operation.element import function
from .util import Dynamic, DynamicMap
from .plotting.util import initialize_dynamic, linear_gradient

_Cmap = Stream.define('Cmap', cmap=[])
_Alpha = Stream.define('Alpha', alpha=1.0)
_Exprs = Stream.define('Exprs', exprs=[])
_Colors = Stream.define('Colors', colors=[])

_SelectionStreams = namedtuple(
    'SelectionStreams', 'colors_stream exprs_stream cmap_streams alpha_streams'
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

        # Init selection streams
        inst._selection_streams = self_or_cls._build_selection_streams(inst)

        return inst

    def _register(self, hvobj):
        """
        Register an Element of DynamicMap that may be capable of generating
        selection expressions in response to user interaction events
        """
        expr_stream = SelectionExpr(source=hvobj)
        expr_stream.add_subscriber(
            lambda **kwargs: self._expr_stream_updated(hvobj, **kwargs)
        )
        self._selection_expr_streams.append(expr_stream)

    def __call__(self, hvobj, **kwargs):
        # Apply kwargs as params
        self.param.set_param(**kwargs)

        # Perform transform
        hvobj_selection = self._selection_transform(hvobj.clone(link=False))

        return hvobj_selection

    def _selection_transform(
            self,
            hvobj,
            operations=(),
    ):
        """
        Transform an input HoloViews object into a dynamic object with linked
        selections enabled.
        """
        if isinstance(hvobj, DynamicMap):
            initialize_dynamic(hvobj)

            if len(hvobj.callback.inputs) == 1 and hvobj.callback.operation:
                child_hvobj = hvobj.callback.inputs[0]
                if isinstance(hvobj.callback, OperationCallable):
                    next_op = hvobj.callback.operation
                else:
                    fn = hvobj.callback.operation
                    next_op = function.instance(fn=fn)
                new_operations = (next_op,) + operations

                # Recurse on child with added operation
                return self._selection_transform(
                    hvobj=child_hvobj,
                    operations=new_operations,
                )
            elif hvobj.type == Overlay and not hvobj.streams:
                # Process overlay inputs individually and then overlay again
                overlay_elements = hvobj.callback.inputs
                new_hvobj = self._selection_transform(overlay_elements[0])
                for overlay_element in overlay_elements[1:]:
                    new_hvobj = new_hvobj * self._selection_transform(overlay_element)

                return new_hvobj
            elif issubclass(hvobj.type, Element):
                self._register(hvobj)

                chart = Store.registry[Store.current_backend][hvobj.type]
                return chart.selection_display.build_selection(
                    self._selection_streams, hvobj, operations
                )
            else:
                # This is a DynamicMap that we don't know how to recurse into.
                return hvobj

        elif isinstance(hvobj, Element):
            element = hvobj.clone(link=False)

            # Register hvobj to receive selection expression callbacks
            self._register(element)

            chart = Store.registry[Store.current_backend][type(element)]
            try:
                return chart.selection_display.build_selection(
                    self._selection_streams, element, operations
                )
            except AttributeError:
                # In case chart doesn't have selection_display defined
                return element

        elif isinstance(hvobj, (Layout, Overlay)):
            new_hvobj = hvobj.clone(shared_data=False)
            for k, v in hvobj.items():
                new_hvobj[k] = self._selection_transform(
                    v, operations
                )

            # collate if available. Needed for Overlay
            try:
                new_hvobj = new_hvobj.collate()
            except AttributeError:
                pass

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

    def _expr_stream_updated(self, hvobj, selection_expr, bbox):
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
    selection_expr = param.Parameter(default=None)
    unselected_color = param.Color(default="#99a6b2")  # LightSlateGray - 65%
    selected_color = param.Color(default="#DC143C")  # Crimson

    @classmethod
    def _build_selection_streams(cls, inst):
        # Colors stream
        colors_stream = _Colors(
            colors=[inst.unselected_color, inst.selected_color]
        )

        # Cmap streams
        cmap_streams = [
            _Cmap(cmap=inst.unselected_cmap),
            _Cmap(cmap=inst.selected_cmap),
        ]

        def update_colors(*_):
            colors_stream.event(
                colors=[inst.unselected_color, inst.selected_color]
            )
            cmap_streams[0].event(cmap=inst.unselected_cmap)
            cmap_streams[1].event(cmap=inst.selected_cmap)

        inst.param.watch(
            update_colors,
            parameter_names=['unselected_color', 'selected_color']
        )

        # Exprs stream
        exprs_stream = _Exprs(exprs=[True, None])

        def update_exprs(*_):
            exprs_stream.event(exprs=[True, inst.selection_expr])

        inst.param.watch(
            update_exprs,
            parameter_names=['selection_expr']
        )

        # Alpha streams
        alpha_streams = [
            _Alpha(alpha=255),
            _Alpha(alpha=inst._selected_alpha),
        ]

        def update_alphas(*_):
            alpha_streams[1].event(alpha=inst._selected_alpha)

        inst.param.watch(update_alphas, parameter_names=['selection_expr'])

        return _SelectionStreams(
            colors_stream=colors_stream,
            exprs_stream=exprs_stream,
            alpha_streams=alpha_streams,
            cmap_streams=cmap_streams,
        )

    @property
    def unselected_cmap(self):
        """
        The datashader colormap for unselected data
        """
        return _color_to_cmap(self.unselected_color)

    @property
    def selected_cmap(self):
        """
        The datashader colormap for selected data
        """
        return _color_to_cmap(self.selected_color)

    @property
    def _selected_alpha(self):
        if self.selection_expr:
            return 255
        else:
            return 0

    def _expr_stream_updated(self, hvobj, selection_expr, bbox):
        if selection_expr:
            self.selection_expr = selection_expr


class SelectionDisplay(object):
    """
    Base class for selection display classes.  Selection display classes are
    responsible for transforming an element (or DynamicMap that produces an
    element) into a HoloViews object that represents the current selection
    state.
    """
    def build_selection(self, selection_streams, hvobj, operations):
        raise NotImplementedError()


class NoOpSelectionDisplay(SelectionDisplay):
    """
    Selection display class that returns input element unchanged. For use with
    elements that don't support displaying selections.
    """
    def build_selection(self, selection_streams, hvobj, operations):
        return hvobj


class OverlaySelectionDisplay(SelectionDisplay):
    """
    Selection display base class that represents selections by overlaying
    colored subsets on top of the original element in an Overlay container.
    """
    def __init__(self, color_prop='color', is_cmap=False):
        self.color_prop = color_prop
        self.is_cmap = is_cmap

    def _get_color_kwarg(self, color):
        return {self.color_prop: [color] if self.is_cmap else color}

    def build_selection(self, selection_streams, hvobj, operations):
        layers = []
        num_layers = len(selection_streams.colors_stream.colors)
        if not num_layers:
            return Overlay(items=[])

        for layer_number in range(num_layers):
            build_layer = self._build_layer_callback(layer_number)
            sel_streams = [selection_streams.colors_stream,
                           selection_streams.exprs_stream]

            if isinstance(hvobj, DynamicMap):
                def apply_map(
                        obj,
                        build_layer=build_layer,
                        colors=None,
                        exprs=None,
                        **kwargs
                ):
                    return obj.map(
                        lambda el: build_layer(el, colors, exprs),
                        specs=Element,
                        clone=True,
                    )

                layer = Dynamic(
                    hvobj,
                    operation=apply_map,
                    streams=hvobj.streams + sel_streams,
                    link_inputs=True,
                )
            else:
                layer = Dynamic(
                        hvobj,
                        operation=build_layer,
                        streams=sel_streams,
                    )

            layers.append(layer)

        # Wrap in operations
        for op in operations:
            for layer_number in range(num_layers):
                streams = copy.copy(op.streams)

                if 'cmap' in op.param:
                    streams += [selection_streams.cmap_streams[layer_number]]

                if 'alpha' in op.param:
                    streams += [selection_streams.alpha_streams[layer_number]]

                new_op = op.instance(streams=streams)
                layers[layer_number] = new_op(layers[layer_number])

        # build overlay
        result = layers[0]
        for layer in layers[1:]:
            result *= layer
        return result

    def _build_layer_callback(self, layer_number):
        def _build_layer(element, colors, exprs, **_):
            layer_element = self._build_element_layer(
                element, colors[layer_number], exprs[layer_number]
            )

            return layer_element

        return _build_layer

    def _build_element_layer(
            self, element, layer_color, selection_expr=True
    ):
        raise NotImplementedError()

    @staticmethod
    def _select(element, selection_expr):
        from .util.transform import dim
        if isinstance(selection_expr, dim):
            try:
                element = element.pipeline(
                    element.dataset.select(selection_expr=selection_expr)
                )
            except Exception as e:
                print(e)
                raise
            visible = True
        else:
            visible = bool(selection_expr)
        return element, visible


class ColorListSelectionDisplay(SelectionDisplay):
    """
    Selection display class for elements that support coloring by a
    vectorized color list.
    """
    def __init__(self, color_prop='color'):
        self.color_prop = color_prop

    def build_selection(self, selection_streams, hvobj, operations):
        def _build_selection(el, colors, exprs, **_):

            selection_exprs = exprs[1:]
            unselected_color = colors[0]
            selected_colors = colors[1:]

            n = len(el.dimension_values(0))

            if not any(selection_exprs):
                colors = [unselected_color] * n
            else:
                clrs = np.array(
                    [unselected_color] + list(selected_colors))

                color_inds = np.zeros(n, dtype='int8')

                for i, expr, color in zip(
                        range(1, len(clrs)),
                        selection_exprs,
                        selected_colors
                ):
                    color_inds[expr.apply(el)] = i

                colors = clrs[color_inds]

            return el.options(**{self.color_prop: colors})

        sel_streams = [selection_streams.colors_stream,
                       selection_streams.exprs_stream]

        if isinstance(hvobj, DynamicMap):
            def apply_map(
                    obj,
                    colors=None,
                    exprs=None,
                    **kwargs
            ):
                return obj.map(
                    lambda el: _build_selection(el, colors, exprs),
                    specs=Element,
                    clone=True,
                )

            hvobj = Dynamic(
                hvobj,
                operation=apply_map,
                streams=hvobj.streams + sel_streams,
                link_inputs=True,
            )
        else:
            hvobj = Dynamic(
                hvobj,
                operation=_build_selection,
                streams=sel_streams
            )

        for op in operations:
            hvobj = op(hvobj)

        return hvobj


def _color_to_cmap(color):
    """
    Create a light to dark cmap list from a base color
    """
    # Lighten start color by interpolating toward white
    start_color = linear_gradient("#ffffff", color, 7)[2]

    # Darken end color by interpolating toward black
    end_color = linear_gradient(color, "#000000", 7)[2]
    return [start_color, end_color]

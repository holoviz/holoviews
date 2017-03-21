"""
Public API for all plots supported by HoloViews, regardless of
plotting package or backend. Every plotting classes must be a subclass
of this Plot baseclass.
"""

from itertools import groupby, product
from collections import Counter, defaultdict

import numpy as np
import param

from ..core import OrderedDict
from ..core import util, traversal
from ..core.element import Element
from ..core.overlay import Overlay, CompositeOverlay
from ..core.layout import Empty, NdLayout, Layout
from ..core.options import Store, Compositor, SkipRendering
from ..core.overlay import NdOverlay
from ..core.spaces import HoloMap, DynamicMap
from ..core.util import stream_parameters
from ..element import Table
from .util import (get_dynamic_mode, initialize_sampled, dim_axis_label,
                   attach_streams, traverse_setter, get_nested_streams)


class Plot(param.Parameterized):
    """
    Base class of all Plot classes in HoloViews, designed to be
    general enough to use any plotting package or backend.
    """

    # A list of style options that may be supplied to the plotting
    # call
    style_opts = []
    # Sometimes matplotlib doesn't support the common aliases.
    # Use this list to disable any invalid style options
    _disabled_opts = []

    def initialize_plot(self, ranges=None):
        """
        Initialize the matplotlib figure.
        """
        raise NotImplementedError


    def update(self, key):
        """
        Update the internal state of the Plot to represent the given
        key tuple (where integers represent frames). Returns this
        state.
        """
        return self.state

    @property
    def state(self):
        """
        The plotting state that gets updated via the update method and
        used by the renderer to generate output.
        """
        raise NotImplementedError


    def __len__(self):
        """
        Returns the total number of available frames.
        """
        raise NotImplementedError

    @classmethod
    def lookup_options(cls, obj, group):
        try:
            plot_class = Store.renderers[cls.backend].plotting_class(obj)
            style_opts = plot_class.style_opts
        except SkipRendering:
            style_opts = None

        node = Store.lookup_options(cls.backend, obj, group)
        if group == 'style' and style_opts:
            return node.filtered(style_opts)
        else:
            return node



class PlotSelector(object):
    """
    Proxy that allows dynamic selection of a plotting class based on a
    function of the plotted object. Behaves like a Plot class and
    presents the same parameterized interface.
    """

    _disabled_opts = []
    def __init__(self, selector, plot_classes, allow_mismatch=False):
        """
        The selector function accepts a component instance and returns
        the appropriate key to index plot_classes dictionary.
        """
        self.selector = selector
        self.plot_classes = OrderedDict(plot_classes)
        interface = self._define_interface(self.plot_classes.values(), allow_mismatch)
        self.style_opts, self.plot_options = interface


    def _define_interface(self, plots, allow_mismatch):
        parameters = [{k:v.precedence for k,v in plot.params().items()
                       if ((v.precedence is None) or (v.precedence >= 0))}
                      for plot in plots]
        param_sets = [set(params.keys()) for params in parameters]
        if not allow_mismatch and not all(pset == param_sets[0] for pset in param_sets):
            raise Exception("All selectable plot classes must have identical plot options.")
        styles= [plot.style_opts for plot in plots]

        if not allow_mismatch and not all(style == styles[0] for style in styles):
            raise Exception("All selectable plot classes must have identical style options.")
        return styles[0], parameters[0]


    def __call__(self, obj, **kwargs):
        key = self.selector(obj)
        if key not in self.plot_classes:
            msg = "Key %s returned by selector not in set: %s"
            raise Exception(msg  % (key, ', '.join(self.plot_classes.keys())))
        return self.plot_classes[key](obj, **kwargs)


    def __setattr__(self, label, value):
        try:
            return super(PlotSelector, self).__setattr__(label, value)
        except:
            raise Exception("Please set class parameters directly on classes %s"
                            % ', '.join(str(cls) for cls in self.__dict__['plot_classes'].values()))

    def params(self):
        return self.plot_options



class DimensionedPlot(Plot):
    """
    DimensionedPlot implements a number of useful methods
    to compute dimension ranges and titles containing the
    dimension values.
    """

    fontsize = param.Parameter(default=None, allow_None=True,  doc="""
       Specifies various fontsizes of the displayed text.

       Finer control is available by supplying a dictionary where any
       unmentioned keys reverts to the default sizes, e.g:

          {'ticks':20, 'title':15,
           'ylabel':5, 'xlabel':5,
           'legend':8, 'legend_title':13}

       You can set the fontsize of both 'ylabel' and 'xlabel' together
       using the 'labels' key.""")

    #Allowed fontsize keys
    _fontsize_keys = ['xlabel','ylabel', 'labels', 'ticks',
                      'title', 'legend', 'legend_title', 'xticks',
                      'yticks']

    show_title = param.Boolean(default=True, doc="""
        Whether to display the plot title.""")

    title_format = param.String(default="{label} {group}\n{dimensions}", doc="""
        The formatting string for the title of this plot, allows defining
        a label group separator and dimension labels.""")

    normalize = param.Boolean(default=True, doc="""
        Whether to compute ranges across all Elements at this level
        of plotting. Allows selecting normalization at different levels
        for nested data containers.""")

    projection = param.Parameter(default=None, doc="""
        Allows supplying a custom projection to transform the axis
        coordinates during display. Example projections include '3d'
        and 'polar' projections supported by some backends. Depending
        on the backend custom projection objects may be supplied.""")

    def __init__(self, keys=None, dimensions=None, layout_dimensions=None,
                 uniform=True, subplot=False, adjoined=None, layout_num=0,
                 style=None, subplots=None, dynamic=False, renderer=None, **params):
        self.subplots = subplots
        self.adjoined = adjoined
        self.dimensions = dimensions
        self.layout_num = layout_num
        self.layout_dimensions = layout_dimensions
        self.subplot = subplot
        self.keys = keys
        self.uniform = uniform
        self.dynamic = dynamic
        self.drawn = False
        self.handles = {}
        self.group = None
        self.label = None
        self.current_frame = None
        self.current_key = None
        self.ranges = {}
        self.renderer = renderer if renderer else Store.renderers[self.backend].instance()
        self.comm = None
        self._force = True

        params = {k: v for k, v in params.items()
                  if k in self.params()}
        super(DimensionedPlot, self).__init__(**params)


    def __getitem__(self, frame):
        """
        Get the state of the Plot for a given frame number.
        """
        if not self.dynamic == 'open' and isinstance(frame, int) and frame > len(self):
            self.warning("Showing last frame available: %d" % len(self))
        if not self.drawn: self.handles['fig'] = self.initialize_plot()
        if not self.dynamic == 'open' and not isinstance(frame, tuple):
            frame = self.keys[frame]
        self.update_frame(frame)
        return self.state


    def _get_frame(self, key):
        """
        Required on each MPLPlot type to get the data corresponding
        just to the current frame out from the object.
        """
        pass


    def matches(self, spec):
        """
        Matches a specification against the current Plot.
        """
        if callable(spec) and not isinstance(spec, type): return spec(self)
        elif isinstance(spec, type): return isinstance(self, spec)
        else:
            raise ValueError("Matching specs have to be either a type or a callable.")


    def traverse(self, fn=None, specs=None, full_breadth=True):
        """
        Traverses any nested DimensionedPlot returning a list
        of all plots that match the specs. The specs should
        be supplied as a list of either Plot types or callables,
        which should return a boolean given the plot class.
        """
        accumulator = []
        matches = specs is None
        if not matches:
            for spec in specs:
                matches = self.matches(spec)
                if matches: break
        if matches:
            accumulator.append(fn(self) if fn else self)

        # Assumes composite objects are iterables
        if hasattr(self, 'subplots') and self.subplots:
            for el in self.subplots.values():
                accumulator += el.traverse(fn, specs, full_breadth)
                if not full_breadth: break
        return accumulator


    def _frame_title(self, key, group_size=2, separator='\n'):
        """
        Returns the formatted dimension group strings
        for a particular frame.
        """
        if self.dynamic == 'open' and self.current_key:
            key = self.current_key
        if self.layout_dimensions is not None:
            dimensions, key = zip(*self.layout_dimensions.items())
        elif not self.dynamic and (not self.uniform or len(self) == 1) or self.subplot:
            return ''
        else:
            key = key if isinstance(key, tuple) else (key,)
            dimensions = self.dimensions
        dimension_labels = [dim.pprint_value_string(k) for dim, k in
                            zip(dimensions, key)]
        groups = [', '.join(dimension_labels[i*group_size:(i+1)*group_size])
                  for i in range(len(dimension_labels))]
        return util.bytes_to_unicode(separator.join(g for g in groups if g))


    def _fontsize(self, key, label='fontsize', common=True):
        if not self.fontsize: return {}

        if not isinstance(self.fontsize, dict):
            return {label:self.fontsize} if common else {}

        unknown_keys = set(self.fontsize.keys()) - set(self._fontsize_keys)
        if unknown_keys:
            msg = "Popping unknown keys %r from fontsize dictionary.\nValid keys: %r"
            self.warning(msg %  (list(unknown_keys), self._fontsize_keys))
            for key in unknown_keys: self.fontsize.pop(key, None)

        if key in self.fontsize:
            return {label:self.fontsize[key]}
        elif key in ['ylabel', 'xlabel'] and 'labels' in self.fontsize:
            return {label:self.fontsize['labels']}
        elif key in ['xticks', 'yticks'] and 'ticks' in self.fontsize:
            return {label:self.fontsize['ticks']}
        else:
            return {}




    def compute_ranges(self, obj, key, ranges):
        """
        Given an object, a specific key and the normalization options
        this method will find the specified normalization options on
        the appropriate OptionTree, group the elements according to
        the selected normalization option (i.e. either per frame or
        over the whole animation) and finally compute the dimension
        ranges in each group. The new set of ranges is returned.
        """
        all_table = all(isinstance(el, Table) for el in obj.traverse(lambda x: x, [Element]))
        if obj is None or not self.normalize or all_table:
            return OrderedDict()
        # Get inherited ranges
        ranges = self.ranges if ranges is None else dict(ranges)

        # Get element identifiers from current object and resolve
        # with selected normalization options
        norm_opts = self._get_norm_opts(obj)

        # Traverse displayed object if normalization applies
        # at this level, and ranges for the group have not
        # been supplied from a composite plot
        return_fn = lambda x: x if isinstance(x, Element) else None
        for group, (axiswise, framewise) in norm_opts.items():
            elements = []
            # Skip if ranges are cached or already computed by a
            # higher-level container object.
            framewise = framewise or self.dynamic or len(elements) == 1
            if group in ranges and (not framewise or ranges is not self.ranges):
                continue
            elif not framewise: # Traverse to get all elements
                elements = obj.traverse(return_fn, [group])
            elif key is not None: # Traverse to get elements for each frame
                frame = self._get_frame(key)
                elements = [] if frame is None else frame.traverse(return_fn, [group])
            # Only compute ranges if not axiswise on a composite plot
            # or not framewise on a Overlay or ElementPlot
            if (not (axiswise and not isinstance(obj, HoloMap)) or
                (not framewise and isinstance(obj, HoloMap))):
                self._compute_group_range(group, elements, ranges)
        self.ranges.update(ranges)
        return ranges


    def _get_norm_opts(self, obj):
        """
        Gets the normalization options for a LabelledData object by
        traversing the object for to find elements and their ids.
        The id is then used to select the appropriate OptionsTree,
        accumulating the normalization options into a dictionary.
        Returns a dictionary of normalization options for each
        element in the tree.
        """
        norm_opts = {}

        # Get all elements' type.group.label specs and ids
        type_val_fn = lambda x: (x.id, (type(x).__name__, util.group_sanitizer(x.group, escape=False),
                                        util.label_sanitizer(x.label, escape=False))) \
            if isinstance(x, Element) else None
        element_specs = {(idspec[0], idspec[1]) for idspec in obj.traverse(type_val_fn)
                         if idspec is not None}

        # Group elements specs by ID and override normalization
        # options sequentially
        key_fn = lambda x: -1 if x[0] is None else x[0]
        id_groups = groupby(sorted(element_specs, key=key_fn), key_fn)
        for gid, element_spec_group in id_groups:
            gid = None if gid == -1 else gid
            group_specs = [el for _, el in element_spec_group]

            backend = self.renderer.backend
            optstree = Store.custom_options(
                backend=backend).get(gid, Store.options(backend=backend))
            # Get the normalization options for the current id
            # and match against customizable elements
            for opts in optstree:
                path = tuple(opts.path.split('.')[1:])
                applies = any(path == spec[:i] for spec in group_specs
                              for i in range(1, 4))
                if applies and 'norm' in opts.groups:
                    nopts = opts['norm'].options
                    if 'axiswise' in nopts or 'framewise' in nopts:
                        norm_opts.update({path: (nopts.get('axiswise', False),
                                                 nopts.get('framewise', False))})
        element_specs = [spec for _, spec in element_specs]
        norm_opts.update({spec: (False, False) for spec in element_specs
                          if not any(spec[:i] in norm_opts.keys() for i in range(1, 4))})
        return norm_opts


    @staticmethod
    def _compute_group_range(group, elements, ranges):
        # Iterate over all elements in a normalization group
        # and accumulate their ranges into the supplied dictionary.
        elements = [el for el in elements if el is not None]
        group_ranges = OrderedDict()
        for el in elements:
            if isinstance(el, (Empty, Table)): continue
            for dim in el.dimensions(label=True):
                dim_range = el.range(dim)
                if dim not in group_ranges:
                    group_ranges[dim] = []
                group_ranges[dim].append(dim_range)
        ranges[group] = OrderedDict((k, util.max_range(v)) for k, v in group_ranges.items())


    @classmethod
    def _traverse_options(cls, obj, opt_type, opts, specs=None, keyfn=None, defaults=True):
        """
        Traverses the supplied object getting all options in opts for
        the specified opt_type and specs. Also takes into account the
        plotting class defaults for plot options. If a keyfn is
        supplied the returned options will be grouped by the returned
        keys.
        """
        def lookup(x):
            """
            Looks up options for object, including plot defaults,
            keyfn determines returned key otherwise None key is used.
            """
            options = cls.lookup_options(x, opt_type)
            selected = {o: options.options[o]
                        for o in opts if o in options.options}
            if opt_type == 'plot' and defaults:
                plot = Store.registry[cls.backend].get(type(x))
                selected['defaults'] = {o: getattr(plot, o) for o in opts
                                        if o not in selected and hasattr(plot, o)}
            key = keyfn(x) if keyfn else None
            return (key, selected)

        # Traverse object and accumulate options by key
        traversed = obj.traverse(lookup, specs)
        options = defaultdict(lambda: defaultdict(list))
        default_opts = defaultdict(lambda: defaultdict(list)) 
        for key, opts in traversed:
            defaults = opts.pop('defaults', {})
            for opt, v in opts.items():
                options[key][opt].append(v)
            for opt, v in defaults.items():
                default_opts[key][opt].append(v)

        # Merge defaults into dictionary if not explicitly specified
        for key, opts in default_opts.items():
            for opt, v in opts.items():
                if opt not in options[key]:
                    options[key][opt] = v
        return options if keyfn else options[None]


    def _get_projection(cls, obj):
        """
        Uses traversal to find the appropriate projection
        for a nested object. Respects projections set on
        Overlays before considering Element based settings,
        before finally looking up the default projection on
        the plot type. If more than one non-None projection
        type is found an exception is raised.
        """
        isoverlay = lambda x: isinstance(x, CompositeOverlay)
        opts = cls._traverse_options(obj, 'plot', ['projection'],
                                     [CompositeOverlay, Element],
                                     keyfn=isoverlay)
        from_overlay = not all(p is None for p in opts[True]['projection'])
        projections = opts[from_overlay]['projection']
        custom_projs = [p for p in projections if p is not None]
        if len(set(custom_projs)) > 1:
            raise Exception("An axis may only be assigned one projection type")
        return custom_projs[0] if custom_projs else None


    def update(self, key):
        if len(self) == 1 and key == 0 and not self.drawn:
            return self.initialize_plot()
        return self.__getitem__(key)


    def refresh(self, **kwargs):
        """
        Refreshes the plot by rerendering it and then pushing
        the updated data if the plot has an associated Comm.
        """
        traverse_setter(self, '_force', True)
        key = self.current_key if self.current_key else self.keys[0]
        dim_streams = [stream for stream in self.streams
                       if any(c in self.dimensions for c in stream.contents)]
        stream_params = stream_parameters(dim_streams)
        key = tuple(None if d in stream_params else k
                    for d, k in zip(self.dimensions, key))
        stream_key = util.wrap_tuple_streams(key, self.dimensions, self.streams)
        self.update(stream_key)
        if self.comm is not None:
            self.push()


    def push(self):
        """
        Pushes updated plot data via the Comm.
        """
        if self.comm is None:
            raise Exception('Renderer does not have a comm.')
        diff = self.renderer.diff(self)
        self.comm.send(diff)


    def init_comm(self, obj):
        """
        Initializes comm and attaches streams.
        """
        comm = None
        if self.dynamic or self.renderer.widget_mode == 'live':
            comm = self.renderer.comms[self.renderer.mode][0](self)
            attach_streams(self, obj)
        return comm


    def __len__(self):
        """
        Returns the total number of available frames.
        """
        return len(self.keys)



class GenericElementPlot(DimensionedPlot):
    """
    Plotting baseclass to render contents of an Element. Implements
    methods to get the correct frame given a HoloMap, axis labels and
    extents and titles.
    """

    apply_ranges = param.Boolean(default=True, doc="""
        Whether to compute the plot bounds from the data itself.""")

    apply_extents = param.Boolean(default=True, doc="""
        Whether to apply extent overrides on the Elements""")

    # A dictionary mapping of the plot methods used to draw the
    # glyphs corresponding to the ElementPlot, can support two
    # keyword arguments a 'single' implementation to draw an individual
    # plot and a 'batched' method to draw multiple Elements at once
    _plot_methods = {}

    # Declares the options that are propagated from sub-elements of the
    # plot, mostly useful for inheriting options from individual
    # Elements on an OverlayPlot. Enabled by default in v1.7.
    _propagate_options = []
    v17_option_propagation = True

    def __init__(self, element, keys=None, ranges=None, dimensions=None,
                 batched=False, overlaid=0, cyclic_index=0, zorder=0, style=None,
                 overlay_dims={}, **params):
        self.zorder = zorder
        self.cyclic_index = cyclic_index
        self.overlaid = overlaid
        self.batched = batched
        self.overlay_dims = overlay_dims

        if not isinstance(element, (HoloMap, DynamicMap)):
            self.hmap = HoloMap(initial_items=(0, element),
                               kdims=['Frame'], id=element.id)
        else:
            self.hmap = element

        plot_element = self.hmap.last
        if self.batched and not isinstance(self, GenericOverlayPlot):
            plot_element = [el for el in plot_element if el][-1]

        top_level = keys is None
        if top_level:
            dimensions = self.hmap.kdims
            keys = list(self.hmap.data.keys())

        self.style = self.lookup_options(plot_element, 'style') if style is None else style
        plot_opts = self.lookup_options(plot_element, 'plot').options
        if self.v17_option_propagation:
            inherited = self._traverse_options(plot_element, 'plot',
                                               self._propagate_options,
                                               defaults=False)
            plot_opts.update(**{k: v[0] for k, v in inherited.items()})

        dynamic = False if not isinstance(element, DynamicMap) or element.sampled else element.mode
        super(GenericElementPlot, self).__init__(keys=keys, dimensions=dimensions,
                                                 dynamic=dynamic,
                                                 **dict(params, **plot_opts))
        if top_level:
            self.comm = self.init_comm(element)
        streams = []
        if isinstance(self.hmap, DynamicMap):
            streams = get_nested_streams(self.hmap)
        self.streams = streams

        # Update plot and style options for batched plots
        if self.batched:
            self.ordering = util.layer_sort(self.hmap)
            overlay_opts = self.lookup_options(self.hmap.last, 'plot').options.items()
            opts = {k: v for k, v in overlay_opts if k in self.params()}
            self.set_param(**opts)
            self.style = self.lookup_options(plot_element, 'style').max_cycles(len(self.ordering))
        else:
            self.ordering = []


    def get_zorder(self, overlay, key, el):
        """
        Computes the z-order of element in the NdOverlay
        taking into account possible batching of elements.
        """
        spec = util.get_overlay_spec(overlay, key, el)
        return self.ordering.index(spec)


    def _updated_zorders(self, overlay):
        specs = [util.get_overlay_spec(overlay, key, el)
                 for key, el in overlay.data.items()]
        self.ordering = sorted(set(self.ordering+specs))
        return [self.ordering.index(spec) for spec in specs]


    def _get_frame(self, key):
        if isinstance(self.hmap, DynamicMap) and self.overlaid and self.current_frame:
            self.current_key = key
            return self.current_frame
        elif self.dynamic:
            key, frame = util.get_dynamic_item(self.hmap, self.dimensions, key)
            traverse_setter(self, '_force', False)
            if not isinstance(key, tuple): key = (key,)
            key_map = dict(zip([d.name for d in self.hmap.kdims], key))
            key = tuple(key_map.get(d.name, None) for d in self.dimensions)
            if not key in self.keys:
                self.keys.append(key)
            self.current_frame = frame
            self.current_key = key
            return frame

        if isinstance(key, int):
            key = list(self.hmap.data.keys())[min([key, len(self.hmap)-1])]

        self.current_key = key

        if self.uniform:
            if not isinstance(key, tuple): key = (key,)
            kdims = [d.name for d in self.hmap.kdims]
            if self.dimensions is None:
                dimensions = kdims
            else:
                dimensions = [d.name for d in self.dimensions]
            if kdims == ['Frame'] and kdims != dimensions:
                select = dict(Frame=0)
            else:
                select = {d: key[dimensions.index(d)]
                          for d in kdims}
        else:
            select = dict(zip(self.hmap.dimensions('key', label=True), key))
        try:
            selection = self.hmap.select((HoloMap, DynamicMap), **select)
        except KeyError:
            selection = None
        selection = selection.last if isinstance(selection, HoloMap) else selection
        self.current_frame = selection

        return selection


    def get_extents(self, view, ranges):
        """
        Gets the extents for the axes from the current View. The globally
        computed ranges can optionally override the extents.
        """
        ndims = len(view.dimensions())
        num = 6 if self.projection == '3d' else 4
        if self.apply_ranges:
            if ranges:
                dims = view.dimensions()
                x0, x1 = ranges[dims[0].name]
                if ndims > 1:
                    y0, y1 = ranges[dims[1].name]
                else:
                    y0, y1 = (np.NaN, np.NaN)
                if self.projection == '3d':
                    if len(dims) > 2:
                        z0, z1 = ranges[dims[2].name]
                    else:
                        z0, z1 = np.NaN, np.NaN
            else:
                x0, x1 = view.range(0)
                y0, y1 = view.range(1) if ndims > 1 else (np.NaN, np.NaN)
                if self.projection == '3d':
                    z0, z1 = view.range(2)
            if self.projection == '3d':
                range_extents = (x0, y0, z0, x1, y1, z1)
            else:
                range_extents = (x0, y0, x1, y1)
        else:
            range_extents = (np.NaN,) * num

        if self.apply_extents:
            norm_opts = self.lookup_options(view, 'norm').options
            if norm_opts.get('framewise', False) or self.dynamic:
                extents = view.extents
            else:
                extent_list = self.hmap.traverse(lambda x: x.extents, [Element])
                extents = util.max_extents(extent_list, self.projection == '3d')
        else:
            extents = (np.NaN,) * num

        if getattr(self, 'shared_axes', False) and self.subplot:
            return util.max_extents([range_extents, extents], self.projection == '3d')
        else:
            max_extent = []
            for l1, l2 in zip(range_extents, extents):
                if (isinstance(l2, util.datetime_types)
                    or (l2 is not None and np.isfinite(l2))):
                    max_extent.append(l2)
                else:
                    max_extent.append(l1)
            return tuple(max_extent)


    def _get_axis_labels(self, dimensions, xlabel=None, ylabel=None, zlabel=None):
        if dimensions and xlabel is None:
            xlabel = dim_axis_label(dimensions[0]) if dimensions[0] else ''
        if len(dimensions) >= 2 and ylabel is None:
            ylabel = dim_axis_label(dimensions[1]) if dimensions[1] else ''
        if self.projection == '3d' and len(dimensions) >= 3 and zlabel is None:
            zlabel = dim_axis_label(dimensions[2]) if dimensions[2] else ''
        return xlabel, ylabel, zlabel


    def _format_title(self, key, separator='\n'):
        frame = self._get_frame(key)
        if frame is None: return None
        type_name = type(frame).__name__
        group = frame.group if frame.group != type_name else ''
        label = frame.label

        dim_title = self._frame_title(key, separator=separator)
        if self.layout_dimensions:
            title = dim_title
        else:
            title_format = util.bytes_to_unicode(self.title_format)
            title = title_format.format(label=util.bytes_to_unicode(label),
                                        group=util.bytes_to_unicode(group),
                                        type=type_name,
                                        dimensions=dim_title)
        return title.strip(' \n')


    def update_frame(self, key, ranges=None):
        """
        Set the plot(s) to the given frame number.  Operates by
        manipulating the matplotlib objects held in the self._handles
        dictionary.

        If n is greater than the number of available frames, update
        using the last available frame.
        """


class GenericOverlayPlot(GenericElementPlot):
    """
    Plotting baseclass to render (Nd)Overlay objects. It implements
    methods to handle the creation of ElementPlots, coordinating style
    groupings and zorder for all layers across a HoloMap. It also
    allows collapsing of layers via the Compositor.
    """

    batched = param.Boolean(default=True, doc="""
        Whether to plot Elements NdOverlay in a batched plotting call
        if possible. Disables legends and zorder may not be preserved.""")

    legend_limit = param.Integer(default=25, doc="""
        Number of rendered glyphs before legends are disabled.""")

    show_legend = param.Boolean(default=True, doc="""
        Whether to show legend for the plot.""")

    style_grouping = param.Integer(default=2,
                                   doc="""The length of the type.group.label
        spec that will be used to group Elements into style groups, i.e.
        a style_grouping value of 1 will group just by type, a value of 2
        will group by type and group and a value of 3 will group by the
        full specification.""")

    _passed_handles = []

    def __init__(self, overlay, ranges=None, batched=True, **params):
        super(GenericOverlayPlot, self).__init__(overlay, ranges=ranges,
                                                 batched=batched, **params)

        # Apply data collapse
        self.hmap = Compositor.collapse(self.hmap, None, mode='data')
        self.hmap = self._apply_compositor(self.hmap, ranges, self.keys)
        self.subplots = self._create_subplots(ranges)
        self.traverse(lambda x: setattr(x, 'comm', self.comm))


    def _apply_compositor(self, holomap, ranges=None, keys=None, dimensions=None):
        """
        Given a HoloMap compute the appropriate (mapwise or framewise)
        ranges in order to apply the Compositor collapse operations in
        display mode (data collapse should already have happened).
        """
        # Compute framewise normalization
        defaultdim = holomap.ndims == 1 and holomap.kdims[0].name != 'Frame'

        if keys and ranges and dimensions and not defaultdim:
            dim_inds = [dimensions.index(d) for d in holomap.kdims]
            sliced_keys = [tuple(k[i] for i in dim_inds) for k in keys]
            frame_ranges = OrderedDict([(slckey, self.compute_ranges(holomap, key, ranges[key]))
                                        for key, slckey in zip(keys, sliced_keys) if slckey in holomap.data.keys()])
        else:
            mapwise_ranges = self.compute_ranges(holomap, None, None)
            frame_ranges = OrderedDict([(key, self.compute_ranges(holomap, key, mapwise_ranges))
                                        for key in holomap.data.keys()])
        ranges = frame_ranges.values()

        return Compositor.collapse(holomap, (ranges, frame_ranges.keys()), mode='display')


    def _create_subplots(self, ranges):
        # Check if plot should be batched
        ordering = util.layer_sort(self.hmap)
        registry = Store.registry[self.renderer.backend]
        batched = self.batched and type(self.hmap.last) is NdOverlay
        if batched:
            batchedplot = registry.get(type(self.hmap.last.last))
        if (batched and batchedplot and 'batched' in batchedplot._plot_methods and
            (not self.show_legend or len(ordering) > self.legend_limit)):
            self.batched = True
            keys, vmaps = [()], [self.hmap]
        else:
            self.batched = False
            keys, vmaps = self.hmap.split_overlays()

        # Compute global ordering
        length = self.style_grouping
        group_fn = lambda x: (x.type.__name__, x.last.group, x.last.label)
        map_lengths = Counter()
        for m in vmaps:
            map_lengths[group_fn(m)[:length]] += 1
        zoffset = 0
        overlay_type = 1 if self.hmap.type == Overlay else 2
        group_counter = Counter()

        subplots = OrderedDict()
        for (key, vmap) in zip(keys, vmaps):
            opts = {'overlaid': overlay_type}
            if self.hmap.type == Overlay:
                style_key = (vmap.type.__name__,) + key
            else:
                if not isinstance(key, tuple): key = (key,)
                style_key = group_fn(vmap) + key
                opts['overlay_dims'] = OrderedDict(zip(self.hmap.last.kdims, key))

            if self.batched:
                vtype = type(vmap.last.last)
                oidx = 0
            else:
                vtype = type(vmap.last)
                oidx = ordering.index(style_key)

            plottype = registry.get(vtype, None)
            if plottype is None:
                self.warning("No plotting class for %s type and %s backend "
                             "found. " % (vtype.__name__, self.renderer.backend))
                continue

            # Get zorder and style counter
            group_key = style_key[:length]
            zorder = self.zorder + oidx + zoffset
            cyclic_index = group_counter[group_key]
            group_counter[group_key] += 1
            group_length = map_lengths[group_key]

            if issubclass(plottype, GenericOverlayPlot):
                opts['show_legend'] = self.show_legend
            elif self.batched and 'batched' in plottype._plot_methods:
                opts['batched'] = self.batched
                opts['overlaid'] = self.overlaid
            if len(ordering) > self.legend_limit:
                opts['show_legend'] = False
            style = self.lookup_options(vmap.last, 'style').max_cycles(group_length)
            passed_handles = {k: v for k, v in self.handles.items()
                              if k in self._passed_handles}
            plotopts = dict(opts, cyclic_index=cyclic_index,
                            invert_axes=self.invert_axes,
                            dimensions=self.dimensions, keys=self.keys,
                            layout_dimensions=self.layout_dimensions,
                            ranges=ranges, show_title=self.show_title,
                            style=style, uniform=self.uniform,
                            zorder=zorder, **passed_handles)

            if not isinstance(key, tuple): key = (key,)
            subplots[key] = plottype(vmap, **plotopts)
            if (not isinstance(plottype, PlotSelector) and
                issubclass(plottype, GenericOverlayPlot)):
                zoffset += len(set([k for o in vmap for k in o.keys()])) - 1
        if not subplots:
            raise SkipRendering("%s backend could not plot any Elements "
                                "in the Overlay." % self.renderer.backend)
        return subplots


    def get_extents(self, overlay, ranges):
        extents = []
        items = overlay.items()
        if self.batched and self.subplots:
            subplot = list(self.subplots.values())[0]
            subplots = [(k, subplot) for k in overlay.data.keys()]
        else:
            subplots = self.subplots.items()
        for key, subplot in subplots:
            found = False
            layer = overlay.data.get(key, None)
            if isinstance(self.hmap, DynamicMap) and layer is None:
                for _, layer in items:
                    if isinstance(layer, subplot.hmap.type):
                        found = True
                        break
                if not found:
                    layer = None
            if layer and subplot.apply_ranges:
                if isinstance(layer, CompositeOverlay):
                    sp_ranges = ranges
                else:
                    sp_ranges = util.match_spec(layer, ranges) if ranges else {}
                extents.append(subplot.get_extents(layer, sp_ranges))
        return util.max_extents(extents, self.projection == '3d')



class GenericCompositePlot(DimensionedPlot):

    def __init__(self, layout, keys=None, dimensions=None, **params):
        if 'uniform' not in params:
            params['uniform'] = traversal.uniform(layout)

        top_level = keys is None
        if top_level:
            dimensions, keys = traversal.unique_dimkeys(layout)

        dynamic, sampled = get_dynamic_mode(layout)
        if sampled:
            initialize_sampled(layout, dimensions, keys[0])
        self.layout = layout
        super(GenericCompositePlot, self).__init__(keys=keys,
                                                   dynamic=dynamic,
                                                   dimensions=dimensions,
                                                   **params)
        if top_level:
            self.comm = self.init_comm(layout)
        self.traverse(lambda x: setattr(x, 'comm', self.comm))
        nested_streams = layout.traverse(lambda x: get_nested_streams(x),
                                         [DynamicMap])
        self.streams = list(set([s for streams in nested_streams for s in streams]))


    def _get_frame(self, key):
        """
        Creates a clone of the Layout with the nth-frame for each
        Element.
        """
        layout_frame = self.layout.clone(shared_data=False)
        keyisint = isinstance(key, int)
        if not isinstance(key, tuple): key = (key,)
        nthkey_fn = lambda x: zip(tuple(x.name for x in x.kdims),
                                  list(x.data.keys())[min([key[0], len(x)-1])])
        if key == self.current_key and not self._force:
            return self.current_frame
        else:
            self.current_key = key

        for path, item in self.layout.items():
            if self.dynamic == 'open':
                if keyisint:
                    counts = item.traverse(lambda x: x.counter, (DynamicMap,))
                    if key[0] >= counts[0]:
                        item.traverse(lambda x: next(x), (DynamicMap,))
                    dim_keys = item.traverse(nthkey_fn, (DynamicMap,))[0]
                else:
                    dim_keys = zip([d.name for d in self.dimensions
                                    if d in item.dimensions('key')], key)
                self.current_key = tuple(k[1] for k in dim_keys)
            elif item.traverse(lambda x: x, [DynamicMap]):
                key, frame = util.get_dynamic_item(item, self.dimensions, key)
                layout_frame[path] = frame
                continue
            elif self.uniform:
                dim_keys = zip([d.name for d in self.dimensions
                                if d in item.dimensions('key')], key)
            else:
                dim_keys = item.traverse(nthkey_fn, (HoloMap,))[0]
            if dim_keys:
                obj = item.select((HoloMap,), **dict(dim_keys))
                if isinstance(obj, HoloMap) and len(obj) == 0:
                    continue
                else:
                    layout_frame[path] = obj
            else:
                layout_frame[path] = item
        traverse_setter(self, '_force', False)

        self.current_frame = layout_frame
        return layout_frame


    def __len__(self):
        return len(self.keys)


    def _format_title(self, key, separator='\n'):
        dim_title = self._frame_title(key, 3, separator)
        layout = self.layout
        type_name = type(self.layout).__name__
        group = util.bytes_to_unicode(layout.group if layout.group != type_name else '')
        label = util.bytes_to_unicode(layout.label)
        title = util.bytes_to_unicode(self.title_format).format(label=label,
                                                                group=group,
                                                                type=type_name,
                                                                dimensions=dim_title)
        return title.strip(' \n')


class GenericLayoutPlot(GenericCompositePlot):
    """
    A GenericLayoutPlot accepts either a Layout or a NdLayout and
    displays the elements in a cartesian grid in scanline order.
    """

    transpose = param.Boolean(default=False, doc="""
        Whether to transpose the layout when plotting. Switches
        from row-based left-to-right and top-to-bottom scanline order
        to column-based top-to-bottom and left-to-right order.""")

    def __init__(self, layout, **params):
        if not isinstance(layout, (NdLayout, Layout)):
            raise ValueError("GenericLayoutPlot only accepts Layout objects.")
        if len(layout.values()) == 0:
            raise SkipRendering(warn=False)

        super(GenericLayoutPlot, self).__init__(layout, **params)
        self.subplots = {}
        self.rows, self.cols = layout.shape[::-1] if self.transpose else layout.shape
        self.coords = list(product(range(self.rows),
                                   range(self.cols)))

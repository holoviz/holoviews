"""
ViewOperations manipulate dataviews, typically for the purposes of
visualization. Such operations often apply to SheetViews or
SheetStacks and compose the data together in ways that can be viewed
conveniently, often by creating or manipulating color channels.
"""

from collections import OrderedDict

import colorsys
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import param
from param import ParamOverrides

from .ndmapping import Dimension
from .views import Overlay, GridLayout, HoloMap
from .sheetviews import SheetView, DataGrid, Contours, VectorField
from .dataviews import View, Items, LayerMap, Table, Items, Curve
from .sheetviews import CoordinateGrid

from .options import options, StyleOpts, ChannelOpts, Cycle
from .styles import GrayNearest

rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)



class ViewOperation(param.ParameterizedFunction):
    """
    A ViewOperation takes one or more views as inputs and processes
    them, returning arbitrary new view objects as output. Individual
    dataviews may be passed in directly while multiple dataviews must
    be passed in as a HoloMap of the appropriate type. A ViewOperation
    may be used to implement simple dataview manipulations or perform
    complex analysis.

    Internally, ViewOperations operate on the level of individual
    dataviews, processing each layer on an input HoloMap independently.
    """

    label = param.String(default='ViewOperation', doc="""
        The label to identify the output of the ViewOperation. By
        default this will match the name of the ViewOperation itself.""")

    def _process(self, view, key=None):
        """
        Process a single input view and output a list of views. When
        multiple views are returned as a list, they will be returned
        to the user as a GridLayout. If a HoloMap is passed into a
        ViewOperation, the individual layers are processed
        sequentially and the dimension keys are passed along with
        the View.
        """
        raise NotImplementedError


    def get_views(self, view, pattern, view_type=SheetView):
        """
        Helper method that return a list of views with labels ending
        with the given pattern and which have the specified type. This
        may be useful to check is a single view satisfies some
        condition or to extract the appropriate views from an Overlay.
        """
        if isinstance(view, Overlay):
            matches = [v for v in view.data if v.label.endswith(pattern)]
        elif isinstance(view, SheetView):
            matches = [view] if view.label.endswith(pattern) else []

        return [match for match in matches if isinstance(match, view_type)]


    def __call__(self, view, **params):
        self.p = ParamOverrides(self, params)

        if isinstance(view, View):
            views = self._process(view)
            if len(views) > 1:
                return GridLayout(views)
            else:
                return views[0]

        elif isinstance(view, CoordinateGrid):
            grids = []
            for pos, cell in view.items():
                val = self(cell, **params)
                stacks = val.values() if isinstance(val, GridLayout) else [val]
                # Initialize the list of data or coordinate grids
                if grids == []:
                    grids = [(DataGrid if not isinstance(stack.type, (SheetLayer))
                              else CoordinateGrid)(view.bounds, None, view.xdensity, view.ydensity, label=view.label)
                             for stack in stacks]
                # Populate the grids
                for ind, stack in enumerate(stacks):
                    grids[ind][pos] = stack

            if len(grids) == 1: return grids[0]
            else:               return GridLayout(grids)


        elif isinstance(view, HoloMap):
            mapped_items = [(k, self._process(k, key=el)) for k, el in view.items()]
            stacks = [LayerMap(dimensions=view.dimensions) for stack_tp in range(len(mapped_items[0][1]))]
            for k, views in mapped_items:
                for ind, v in enumerate(views):
                    stacks[ind][k] = v

            if len(stacks) == 1:  return stacks[0]
            else:                 return GridLayout(stacks)



class StackOperation(param.ParameterizedFunction):
    """
    A StackOperation takes a HoloMap of Views or Overlays as inputs
    and processes them, returning arbitrary new HoloMap objects as output.
    """

    label = param.String(default='StackOperation', doc="""
        The label to identifiy the output of the StackOperation. By
        default this will match the name of the StackOperation.""")


    def __call__(self, stack, **params):
        self.p = ParamOverrides(self, params)

        if not isinstance(stack, HoloMap):
            raise Exception('StackOperation can only process Stacks.')

        stacks = self._process(stack)

        if len(stacks) == 1:
            return stacks[0]
        else:
            return GridLayout(stacks)


    def _process(self, view):
        """
        Process a single input HoloMap and output a list of views or
        stacks. When multiple values are returned they are returned to
        the user as a GridLayout.
        """
        raise NotImplementedError


class chain(ViewOperation):
    """
    Definining a viewoperation chain is an easy way to define a new
    ViewOperation from a series of existing ones. The single argument
    is a callable that accepts an input view and returns a list of
    output views. To create the custom ViewOperation, you will need to
    supply this argument to a new instance of chain. For example:

    chain.instance(chain=lambda x: [cmap2rgb(operator(x).N, cmap='jet')])

    This is now a ViewOperation that sums the data in the input
    overlay and turns it into an RGB SheetView with the 'jet'
    colormap.
    """

    chain = param.Callable(doc="""A chain of existing ViewOperations.""")

    def _process(self, view, key=None):
        return self.p.chain(view)



class operator(ViewOperation):
    """
    Applies any arbitrary operator on the data (currently only
    supports SheetViews) and returns the result.
    """

    operator = param.Callable(np.add, doc="""
        The binary operator to apply between the data attributes of
        the supplied Views. By default, performs elementwise addition
        across the SheetView arrays.""")

    label = param.String(default='Operation', doc="""
        The label for the result after having applied the operator.""")

    def _process(self, overlay, key=None):

        if not isinstance(overlay, Overlay):
            raise Exception("Operation requires an Overlay as input")

        new_data = self.p.operator(*[el.data for el in overlay.data])
        return [SheetView(new_data, bounds=overlay.bounds, label=self.p.label,
                          roi_bounds=overlay.roi_bounds)]


class RGBA(ViewOperation):
    """
    Accepts an overlay containing either 3 or 4 layers. The first
    three layers are the R,G, B channels and the last layer (if given)
    is the alpha channel.
    """

    label = param.String(default='RGBA', doc="""
        The label to use for the resulting RGBA SheetView.""")

    def _process(self, overlay, key=None):
        if len(overlay) not in [3, 4]:
            raise Exception("Requires 3 or 4 layers to convert to RGB(A)")
        if not all(isinstance(el, SheetView) for el in overlay.data):
            raise Exception("All layers must be SheetViews to convert"
                            " to RGB(A) format")
        if not all(el.depth == 1 for el in overlay.data):
            raise Exception("All SheetViews must have a depth of one for"
                            " conversion to RGB(A) format")

        arrays = []
        for el in overlay.data:
            if el.data.max() > 1.0 or el.data.min() < 0:
                self.warning("Clipping data into the interval [0, 1]")
                el.data.clip(0,1.0)
            arrays.append(el.data)


        return [SheetView(np.dstack(arrays), overlay.data[0].bounds, label=self.p.label,
                          roi_bounds=overlay.data[0].roi_bounds, value=overlay[0].value)]


class alpha_overlay(ViewOperation):
    """
    Accepts an overlay of a SheetView defined with a cmap and converts
    it to an RGBA SheetView. The alpha channel of the result is
    defined by the second layer of the overlay.
    """

    label = param.String(default='AlphaOverlay', doc="""
        The label suffix to use for the alpha overlay result where the
        suffix is added to the label of the first layer.""")

    def _process(self, overlay, key=None):
        R,G,B,_ = split(cmap2rgb(overlay[0]))
        return [SheetView(RGBA(R*G*B*overlay[1]).data, overlay[0].bounds,
                          label=self.p.label, value=overlay[0].value)]



class HCS(ViewOperation):
    """
    Hue-Confidence-Strength plot.

    Accepts an overlay containing either 2 or 3 layers. The first two
    layers are hue and confidence and the third layer (if available)
    is the strength channel.
    """

    S_multiplier = param.Number(default=1.0, bounds=(0.0,None), doc="""
        Multiplier for the strength value.""")

    C_multiplier = param.Number(default=1.0, bounds=(0.0,None), doc="""
        Multiplier for the confidence value.""")

    flipSC = param.Boolean(default=False, doc="""
        Whether to flip the strength and confidence channels""")

    label = param.String(default='HCS', doc="""
        The label suffix to use for the resulting HCS plot where the
        suffix is added to the label of the Hue channel.""")

    def _process(self, overlay, key=None):
        hue = overlay[0]
        confidence = overlay[1]

        strength_data = overlay[2].data if (len(overlay) == 3) else np.ones(hue.shape)

        if hue.shape != confidence.shape:
            raise Exception("Cannot combine plots with different shapes")

        (h,s,v)= (hue.N.data.clip(0.0, 1.0),
                  (confidence.data * self.p.C_multiplier).clip(0.0, 1.0),
                  (strength_data * self.p.S_multiplier).clip(0.0, 1.0))

        if self.p.flipSC:
            (h,s,v) = (h,v,s.clip(0,1.0))

        r, g, b = hsv_to_rgb(h, s, v)
        rgb = np.dstack([r,g,b])
        return [SheetView(rgb, hue.bounds, roi_bounds=hue.roi_bounds,
                          label=self.p.label, value=hue.value)]



class colorize(ViewOperation):
    """
    Given a Overlay consisting of a grayscale colormap and a
    second Sheetview with some specified colour map, use the second
    layer to colorize the data of the first layer.

    Currently, colorize only support the 'hsv' color map and is just a
    shortcut to the HCS operation using a constant confidence
    value. Arbitrary colorization will be supported in future.
    """

    label = param.String(default='Colorized', doc="""
        The label suffix to use for the resulting colorized plot where
        the suffix is added to the label of the first layer.""")

    def _process(self, overlay, key=None):

         if len(overlay) != 2 and overlay[0].mode != 'cmap':
             raise Exception("Can only colorize grayscale overlayed with colour map.")
         if [overlay[0].depth, overlay[1].depth ] != [1,1]:
             raise Exception("Depth one layers required.")
         if overlay[0].shape != overlay[1].shape:
             raise Exception("Shapes don't match.")

         # Needs a general approach which works with any color map
         C = SheetView(np.ones(overlay[0].data.shape),
                       bounds=overlay[0].bounds)
         hcs = HCS(overlay[1] * C * overlay[0].N)

         return [SheetView(hcs.data, hcs.bounds, roi_bounds=hcs.roi_bounds,
                           label=self.p.label, value=hcs.value)]



class cmap2rgb(ViewOperation):
    """
    Convert SheetViews using colormaps to RGBA mode.  The colormap of
    the style is used, if available. Otherwise, the colormap may be
    forced as a parameter.
    """

    cmap = param.String(default=None, allow_None=True, doc="""
          Force the use of a specific color map. Otherwise, the cmap
          property of the applicable style is used.""")

    label = param.String(default='RGB', doc="""
        The label suffix to use for the resulting RGB SheetView where
        the suffix is added to the label of the SheetView to be
        colored.""")

    def _process(self, sheetview, key=None):
        if sheetview.depth != 1:
            raise Exception("Can only apply colour maps to SheetViews with depth of 1.")

        style_cmap = options.style(sheetview)[0].get('cmap', None)
        if not any([self.p.cmap, style_cmap]):
            raise Exception("No color map supplied and no cmap in the active style.")

        cmap = matplotlib.cm.get_cmap(style_cmap if self.p.cmap is None else self.p.cmap)
        return [sheetview.clone(cmap(sheetview.data), label=self.p.label)]



class split(ViewOperation):
    """
    Given SheetViews in RGBA mode, return the R,G,B and A channels as
    a GridLayout.
    """

    label = param.String(default='Channel', doc="""
      The label suffix used to label the components of the split
      following the character selected from output_names.""")

    def _process(self, sheetview, key=None):
        if sheetview.mode not in ['rgb', 'rgba']:
            raise Exception("Can only split SheetViews with a depth of 3 or 4")
        return [sheetview.clone(sheetview.data[:, :, i],
                                label='RGBA'[i] + ' ' + self.p.label)
                for i in range(sheetview.depth)]




class contours(ViewOperation):
    """
    Given a SheetView with a single channel, annotate it with contour
    lines for a given set of contour levels.

    The return is a overlay with a Contours layer for each given
    level, overlaid on top of the input SheetView.
    """

    levels = param.NumericTuple(default=(0.5,), doc="""
         A list of scalar values used to specify the contour levels.""")

    label = param.String(default='Level', doc="""
      The label suffix used to label the resulting contour curves
      where the suffix is added to the label of the  input SheetView""")

    def _process(self, sheetview, key=None):

        figure_handle = plt.figure()
        (l, b, r, t) = sheetview.bounds.lbrt()
        contour_set = plt.contour(sheetview.data, extent=(l, r, t, b),
                                  levels=self.p.levels)

        contours = []
        for level, cset in zip(self.p.levels, contour_set.collections):
            paths = cset.get_paths()
            lines = [path.vertices for path in paths]
            contours.append(Contours(lines, label=sheetview.label + ' ' + self.p.label))

        plt.close(figure_handle)

        if len(contours) == 1:
            return [(sheetview * contours[0])]
        else:
            return [sheetview * Overlay(contours, sheetview.bounds)]


class vectorfield(ViewOperation):
    """
    Given a SheetView with a single channel, convert it to a
    VectorField object at a given spatial sampling interval. The
    values in the SheetView are assumed to correspond to the vector
    angle in radians and the value is assumed to be cyclic.

    If supplied with an overlay, the second sheetview in the overlay
    will be interpreted as the third vector dimension.
    """

    rows = param.Integer(default=10, doc="""
         Number of rows in the vector field.""")

    cols = param.Integer(default=10, doc="""
         Number of columns in the vector field.""")

    label = param.String(default='Vectors', doc="""
      The label suffix used to label the resulting vector field
      where the suffix is added to the label of the  input SheetView""")


    def _process(self, view, key=None):

        if isinstance(view, Overlay) and len(view) >= 2:
            radians, lengths = view[0], view[1]
        else:
            radians, lengths = view, None

        if not radians.value.cyclic:
            raise Exception("First input SheetView must be declared cyclic")

        l, b, r, t = radians.bounds.lbrt()
        X, Y = np.meshgrid(np.linspace(l, r, self.p.cols+2)[1:-1],
                           np.linspace(b, t, self.p.rows+2)[1:-1])

        vector_data = []
        for x, y in zip(X.flat, Y.flat):

            components = (x,y, radians[x,y])
            if lengths is not None:
                components += (lengths[x,y],)

            vector_data.append(components)

        value_dimension = Dimension('VectorField', cyclic=True, range=radians.range)
        return [VectorField(vector_data,
                            label=radians.label + ' ' + self.p.label,
                            value=value_dimension)]



class threshold(ViewOperation):
    """
    Threshold a given SheetView at a given level into the specified
    low and high values.  """

    level = param.Number(default=0.5, doc="""
       The value at which the threshold is applied. Values lower than
       the threshold map to the 'low' value and values above map to
       the 'high' value.""")

    high = param.Number(default=1.0, doc="""
      The value given to elements greater than (or equal to) the
      threshold.""")

    low = param.Number(default=0.0, doc="""
      The value given to elements below the threshold.""")

    label = param.String(default='Thresholded', doc="""
       The label suffix used to label the resulting sheetview where
       the suffix is added to the label of the input SheetView""")

    def _process(self, view, key=None):
        arr = view.data
        high = np.ones(arr.shape) * self.p.high
        low = np.ones(arr.shape) * self.p.low
        thresholded = np.where(arr > self.p.level, high, low)

        return [SheetView(thresholded,
                          label=view.label + ' ' + self.p.label)]



class roi_table(ViewOperation):
    """
    Compute a table of information from a SheetView within the
    indicated region-of-interest (ROI). The function applied must
    accept a numpy array and return either a single value or a
    dictionary of values which is returned as a Items or TableStack.

    The roi is specified using a boolean SheetView overlay (e.g as
    generated by threshold) where zero values are excluded from the
    ROI.
    """

    fn = param.Callable(default=np.mean, doc="""
        The function that is applied within the mask area to supply
        the relevant information. Other valid examples include np.sum,
        np.median, np.std etc.""")

    heading = param.String(default='result', doc="""
       If the output of the function is not a dictionary, this is the
       label given to the resulting value.""")

    label = param.String(default='ROI', doc="""
       The label suffix that labels the resulting table where this
       suffix is added to the label of the input data SheetView""")


    def _process(self, view, key=None):

        if not isinstance(view, Overlay) or len(view) != 2:
            raise Exception("A Overlay of two SheetViews is required.")

        sview, mask = view[0], view[1]

        if sview.data.shape != mask.data.shape:
            raise Exception("The data array shape of the mask layer"
                            " must match that of the data layer.")

        bit_mask = mask.data.astype(np.bool)
        roi = sview.data[bit_mask]

        results = self.p.fn(roi)
        if not isinstance(results, dict):
            results = {self.p.heading:results}

        return [Items(results, label=sview.label + ' ' + self.p.label)]


class TableCollate(StackOperation):

    collation_dim = param.String(default="")

    def _process(self, stack):
        collate_dim = self.p.collation_dim
        new_dimensions = [d for d in stack.dimensions if d.name != collate_dim]
        nested_stack = stack.split_dimensions([collate_dim]) if new_dimensions else {(): stack}
        collate_dim = stack.dim_dict[collate_dim]
        
        table = stack.last
        table_dims = table.dimensions
        if isinstance(stack.last, Table):
            outer_dims = table_dims[-2:]
            new_dimensions += [td for td in table_dims if td not in outer_dims] 
            entry_keys = [k[-2:] for k in table.data.keys()]
        else:
            outer_dims = ['Label']
            entry_keys = table.data.keys()

        # Generate a LayerMap for every entry in the table
        stack_fn = lambda: LayerMap(**dict(stack.get_param_values(), dimensions=new_dimensions))
        entries = [(entry, (stack_fn() if new_dimensions else None)) for entry in entry_keys]
        stacks = HoloMap(entries, dimensions=outer_dims)
        for new_key, collate_stack in nested_stack.items():
            curve_data = OrderedDict([(k, []) for k in entry_keys])
            # Get the x- and y-values for each entry in the Items
            xvalues = [float(k) for k in collate_stack.keys()]
            for x, table in collate_stack.items():
                for label, value in table.data.items():
                    entry_key = label[-2:] if isinstance(table, Table) else label
                    curve_data[entry_key].append(float(value))

            # Get data from table
            table = collate_stack.last
            table_dimensions = table.dimensions
            table_title = ' ' + table.title
            table_label = table.label
            table_value = table.value

            # Generate curves with correct dimensions
            for label, yvalues in curve_data.items():
                settings = dict(dimensions=[collate_dim])
                if isinstance(table, Table):
                    if not isinstance(label, tuple): label = (label,)
                    if not isinstance(new_key, tuple): new_key = (new_key,)
                    settings.update(value=table.value, label=table_label,
                                    dimensions=[collate_dim])
                    key = new_key + label[0:max(0,len(label)-1)]
                    label = label[-2:]
                else:
                    key = new_key
                    value = table.dim_dict[label]
                    settings.update(value=value, label=table_label,
                                    title='{label} - {value}')
                curve = Curve(zip(xvalues, yvalues), **settings)
                if new_dimensions:
                    stacks[label][key] = curve
                else:
                    stacks[label] = curve

        # If there are multiple table entries, generate grid
        if stacks.ndims in [1, 2]:
            stack = stacks.map(lambda x,k: x.last)
        if isinstance(table, Table):
            grid = stacks.grid(stacks.dimension_labels)
        else:
            grid = stacks.grid(['Label'], layout=True, constant_dims=False)
        return [grid] 



ChannelOpts.operations['RGBA'] = RGBA
ChannelOpts.operations['HCS'] = HCS
ChannelOpts.operations['alpha_overlay'] = alpha_overlay

options.R_Channel_SheetView = GrayNearest
options.G_Channel_SheetView = GrayNearest
options.B_Channel_SheetView = GrayNearest
options.A_Channel_SheetView = GrayNearest
options.Level_Contours = StyleOpts(color=Cycle(['b', 'g', 'r']))

options.RGB_SheetView = StyleOpts(interpolation='nearest')
options.RGBA_SheetView = StyleOpts(interpolation='nearest')

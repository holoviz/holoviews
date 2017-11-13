from itertools import groupby
import numpy as np

import param

from .dimension import Dimension, Dimensioned, ViewableElement
from .layout import Composable, Layout, NdLayout
from .ndmapping import OrderedDict, NdMapping
from .overlay import Overlayable, NdOverlay, CompositeOverlay
from .spaces import HoloMap, GridSpace
from .tree import AttrTree
from .util import get_param_values


class Element(ViewableElement, Composable, Overlayable):
    """
    Element is the baseclass for all ViewableElement types, with an x- and
    y-dimension. Subclasses should define the data storage in the
    constructor, as well as methods and properties, which define how
    the data maps onto the x- and y- and value dimensions.
    """

    group = param.String(default='Element', constant=True)

    def hist(self, dimension=None, num_bins=20, bin_range=None,
             adjoin=True, individually=True, **kwargs):
        """
        The hist method generates a histogram to be adjoined to the
        Element in an AdjointLayout. By default the histogram is
        computed along the first value dimension of the Element,
        however any dimension may be selected. The number of bins and
        the bin_ranges and any kwargs to be passed to the histogram
        operation may also be supplied.
        """
        from ..operation import histogram
        if not isinstance(dimension, list): dimension = [dimension]
        hists = []
        for d in dimension[::-1]:
            hist = histogram(self, num_bins=num_bins, bin_range=bin_range,
                             individually=individually, dimension=d, **kwargs)
            hists.append(hist)
        if adjoin:
            layout = self
            for didx in range(len(dimension)):
                layout = layout << hists[didx]
        elif len(dimension) > 1:
            layout = Layout(hists)
        else:
            layout = hists[0]
        return layout

    #======================#
    # Subclassable methods #
    #======================#


    def __getitem__(self, key):
        if key is ():
            return self
        else:
            raise NotImplementedError("%s currently does not support getitem" %
                                      type(self).__name__)

    def __nonzero__(self):
        """
        Subclasses may override this to signal that the Element contains
        no data and can safely be dropped during indexing.
        """
        return True

    __bool__ = __nonzero__


    @classmethod
    def collapse_data(cls, data, function=None, kdims=None, **kwargs):
        """
        Class method to collapse a list of data matching the
        data format of the Element type. By implementing this
        method HoloMap can collapse multiple Elements of the
        same type. The kwargs are passed to the collapse
        function. The collapse function must support the numpy
        style axis selection. Valid function include:
        np.mean, np.sum, np.product, np.std, scipy.stats.kurtosis etc.
        Some data backends also require the key dimensions
        to aggregate over.
        """
        raise NotImplementedError("Collapsing not implemented for %s." % cls.__name__)


    def closest(self, coords):
        """
        Class method that returns the exact keys for a given list of
        coordinates. The supplied bounds defines the extent within
        which the samples are drawn and the optional shape argument is
        the shape of the numpy array (typically the shape of the .data
        attribute) when applicable.
        """
        return coords


    def sample(self, samples=[], **sample_values):
        """
        Base class signature to demonstrate API for sampling Elements.
        To sample an Element supply either a list of samples or keyword
        arguments, where the key should match an existing key dimension
        on the Element.
        """
        raise NotImplementedError


    def reduce(self, dimensions=[], function=None, **reduce_map):
        """
        Base class signature to demonstrate API for reducing Elements,
        using some reduce function, e.g. np.mean, which is applied
        along a particular Dimension. The dimensions and reduce functions
        should be passed as keyword arguments or as a list of dimensions
        and a single function.
        """
        raise NotImplementedError


    def _reduce_map(self, dimensions, function, reduce_map):
        if dimensions and reduce_map:
            raise Exception("Pass reduced dimensions either as an argument "
                            "or as part of the kwargs not both.")
        if len(set(reduce_map.values())) > 1:
            raise Exception("Cannot define reduce operations with more than "
                            "one function at a time.")
        if reduce_map:
            reduce_map = reduce_map.items()
        if dimensions:
            reduce_map = [(d, function) for d in dimensions]
        elif not reduce_map:
            reduce_map = [(d, function) for d in self.kdims]
        reduced = [(self.get_dimension(d, strict=True).name, fn)
                   for d, fn in reduce_map]
        grouped = [(fn, [dim for dim, _ in grp]) for fn, grp in groupby(reduced, lambda x: x[1])]
        return grouped[0]


    def table(self, datatype=None):
        """
        Converts the data Element to a Table, optionally may
        specify a supported data type. The default data types
        are 'numpy' (for homogeneous data), 'dataframe', and
        'dictionary'.
        """
        if datatype and not isinstance(datatype, list):
            datatype = [datatype]
        from ..element import Table
        return Table(self, **(dict(datatype=datatype) if datatype else {}))


    def dframe(self, dimensions=None):
        import pandas as pd
        column_names = dimensions if dimensions else self.dimensions(label=True)
        dim_vals = OrderedDict([(dim, self[dim]) for dim in column_names])
        return pd.DataFrame(dim_vals)


    def mapping(self, kdims=None, vdims=None, **kwargs):
        length = len(self)
        if not kdims: kdims = self.kdims
        if kdims:
            keys = zip(*[self.dimension_values(dim.name)
                         for dim in self.kdims])
        else:
            keys = [()]*length

        if not vdims: vdims = self.vdims
        if vdims:
            values = zip(*[self.dimension_values(dim.name)
                           for dim in vdims])
        else:
            values = [()]*length
        return OrderedDict(zip(keys, values))


    def array(self, dimensions=[]):
        if dimensions:
            dims = [self.get_dimension(d, strict=True) for d in dimensions]
        else:
            dims = [d for d in self.kdims + self.vdims if d != 'Index']
        columns, types = [], []
        for dim in dims:
            column = self.dimension_values(dim)
            columns.append(column)
            types.append(column.dtype.kind)
        if len(set(types)) > 1:
            columns = [c.astype('object') for c in columns]
        return np.column_stack(columns)



class Tabular(Element):
    """
    Baseclass to give an NdMapping objects an API to generate a
    table representation.
    """

    __abstract = True

    @property
    def rows(self):
        return len(self) + 1

    @property
    def cols(self):
        return len(self.dimensions())


    def pprint_cell(self, row, col):
        """
        Get the formatted cell value for the given row and column indices.
        """
        ndims = self.ndims
        if col >= self.cols:
            raise Exception("Maximum column index is %d" % self.cols-1)
        elif row >= self.rows:
            raise Exception("Maximum row index is %d" % self.rows-1)
        elif row == 0:
            if col >= ndims:
                if self.vdims:
                    return self.vdims[col - ndims].pprint_label
                else:
                    return ''
            return self.kdims[col].pprint_label
        else:
            dim = self.get_dimension(col)
            return dim.pprint_value(self.iloc[row-1, col])


    def cell_type(self, row, col):
        """
        Returns the cell type given a row and column index. The common
        basic cell types are 'data' and 'heading'.
        """
        return 'heading' if row == 0 else 'data'



class Element2D(Element):

    extents = param.Tuple(default=(None, None, None, None),
                          doc="""Allows overriding the extents
              of the Element in 2D space defined as four-tuple
              defining the (left, bottom, right and top) edges.""")


class Element3D(Element2D):

    extents = param.Tuple(default=(None, None, None,
                                   None, None, None),
        doc="""Allows overriding the extents of the Element
               in 3D space defined as (xmin, ymin, zmin,
               xmax, ymax, zmax).""")


class Collator(NdMapping):
    """
    Collator is an NdMapping type which can merge any number
    of HoloViews components with whatever level of nesting
    by inserting the Collators key dimensions on the HoloMaps.
    If the items in the Collator do not contain HoloMaps
    they will be created. Collator also supports filtering
    of Tree structures and dropping of constant dimensions.
    """

    drop = param.List(default=[], doc="""
        List of dimensions to drop when collating data, specified
        as strings.""")

    drop_constant = param.Boolean(default=False, doc="""
        Whether to demote any non-varying key dimensions to
        constant dimensions.""")

    filters = param.List(default=[], doc="""
        List of paths to drop when collating data, specified
        as strings or tuples.""")

    group = param.String(default='Collator')

    progress_bar = param.Parameter(default=None, doc="""
         The progress bar instance used to report progress. Set to
         None to disable progress bars.""")

    merge_type = param.ClassSelector(class_=NdMapping, default=HoloMap,
                                     is_instance=False,instantiate=False)

    value_transform = param.Callable(default=None, doc="""
        If supplied the function will be applied on each Collator
        value during collation. This may be used to apply an operation
        to the data or load references from disk before they are collated
        into a displayable HoloViews object.""")

    vdims = param.List(default=[], doc="""
         Collator operates on HoloViews objects, if vdims are specified
         a value_transform function must also be supplied.""")

    _deep_indexable = False
    _auxiliary_component = False

    _nest_order = {HoloMap: ViewableElement,
                   GridSpace: (HoloMap, CompositeOverlay, ViewableElement),
                   NdLayout: (GridSpace, HoloMap, ViewableElement),
                   NdOverlay: Element}

    def __init__(self, data=None, **params):
        if isinstance(data, Element):
            params = dict(get_param_values(data), **params)
            if 'kdims' not in params:
                params['kdims'] = data.kdims
            if 'vdims' not in params:
                params['vdims'] = data.vdims
            data = data.mapping()
        super(Collator, self).__init__(data, **params)


    def __call__(self):
        """
        Filter each Layout in the Collator with the supplied
        path_filters. If merge is set to True all Layouts are
        merged, otherwise an NdMapping containing all the
        Layouts is returned. Optionally a list of dimensions
        to be ignored can be supplied.
        """
        constant_dims = self.static_dimensions
        ndmapping = NdMapping(kdims=self.kdims)

        num_elements = len(self)
        for idx, (key, data) in enumerate(self.data.items()):
            if isinstance(data, AttrTree):
                data = data.filter(self.filters)
            if len(self.vdims) and self.value_transform:
                vargs = dict(zip(self.dimensions('value', label=True), data))
                data = self.value_transform(vargs)
            if not isinstance(data, Dimensioned):
                raise ValueError("Collator values must be Dimensioned objects "
                                 "before collation.")

            dim_keys = zip(self.kdims, key)
            varying_keys = [(d, k) for d, k in dim_keys if not self.drop_constant or
                            (d not in constant_dims and d not in self.drop)]
            constant_keys = [(d, k) for d, k in dim_keys if d in constant_dims
                             and d not in self.drop and self.drop_constant]
            if varying_keys or constant_keys:
                data = self._add_dimensions(data, varying_keys,
                                            dict(constant_keys))
            ndmapping[key] = data
            if self.progress_bar is not None:
                self.progress_bar(float(idx+1)/num_elements*100)

        components = ndmapping.values()
        accumulator = ndmapping.last.clone(components[0].data)
        for component in components:
            accumulator.update(component)
        return accumulator


    @property
    def static_dimensions(self):
        """
        Return all constant dimensions.
        """
        dimensions = []
        for dim in self.kdims:
            if len(set(self.dimension_values(dim.name))) == 1:
                dimensions.append(dim)
        return dimensions


    def _add_dimensions(self, item, dims, constant_keys):
        """
        Recursively descend through an Layout and NdMapping objects
        in order to add the supplied dimension values to all contained
        HoloMaps.
        """
        if isinstance(item, Layout):
            item.fixed = False

        dim_vals = [(dim, val) for dim, val in dims[::-1]
                    if dim not in self.drop]
        if isinstance(item, self.merge_type):
            new_item = item.clone(cdims=constant_keys)
            for dim, val in dim_vals:
                dim = dim if isinstance(dim, Dimension) else Dimension(dim)
                if dim not in new_item.kdims:
                    new_item = new_item.add_dimension(dim, 0, val)
        elif isinstance(item, self._nest_order[self.merge_type]):
            if len(dim_vals):
                dimensions, key = zip(*dim_vals)
                new_item = self.merge_type({key: item}, kdims=list(dimensions),
                                           cdims=constant_keys)
            else:
                new_item = item
        else:
            new_item = item.clone(shared_data=False, cdims=constant_keys)
            for k, v in item.items():
                new_item[k] = self._add_dimensions(v, dims[::-1], constant_keys)
        if isinstance(new_item, Layout):
            new_item.fixed = True

        return new_item


__all__ = list(set([_k for _k, _v in locals().items()
                    if isinstance(_v, type) and issubclass(_v, Dimensioned)]))

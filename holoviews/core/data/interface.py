import param
import numpy as np

from ..element import Element, NdElement
from .. import util


class Interface(param.Parameterized):

    interfaces = {}

    datatype = None

    @classmethod
    def register(cls, interface):
        cls.interfaces[interface.datatype] = interface


    @classmethod
    def cast(cls, dataset, datatype=None, cast_type=None):
        """
        Given a list of Dataset objects, cast them to the specified
        datatype (by default the format matching the current interface)
        with the given cast_type (if specified).
        """
        if len({type(c) for c in dataset}) > 1 and cast_type is None:
            raise Exception("Please supply the common cast type")

        if datatype is None:
           datatype = cls.datatype

        unchanged = all({c.interface==cls for c in dataset})
        if unchanged and cast_type is None:
            return dataset
        elif unchanged:
            return [cast_type(co, **dict(util.get_param_values(co)))
                    for co in dataset]

        return [co.clone(co.columns(), datatype=[datatype], new_type=cast_type)
                for co in dataset]


    @classmethod
    def initialize(cls, eltype, data, kdims, vdims, datatype=None):
        # Process params and dimensions
        if isinstance(data, Element):
            pvals = util.get_param_values(data)
            kdims = pvals.get('kdims') if kdims is None else kdims
            vdims = pvals.get('vdims') if vdims is None else vdims

        # Process Element data
        if isinstance(data, NdElement):
            kdims = [kdim for kdim in kdims if kdim != 'Index']
        elif hasattr(data, 'interface') and issubclass(data.interface, Interface):
            data = data.data
        elif isinstance(data, Element):
            data = tuple(data.dimension_values(d) for d in kdims+vdims)
        elif isinstance(data, util.generator_types):
            data = list(data)

        # Set interface priority order
        if datatype is None:
            datatype = eltype.datatype
        prioritized = [cls.interfaces[p] for p in datatype
                       if p in cls.interfaces]

        head = [intfc for intfc in prioritized if type(data) in intfc.types]
        if head:
            # Prioritize interfaces which have matching types
            prioritized = head + [el for el in prioritized if el != head[0]]

        # Iterate over interfaces until one can interpret the input
        for interface in prioritized:
            try:
                (data, dims, extra_kws) = interface.init(eltype, data, kdims, vdims)
                break
            except:
                pass
        else:
            raise ValueError("None of the available storage backends "
                             "were able to support the supplied data format.")

        return data, interface, dims, extra_kws


    @classmethod
    def validate(cls, dataset):
        not_found = [d for d in dataset.dimensions(label=True)
                     if d not in dataset.data]
        if not_found:
            raise ValueError("Supplied data does not contain specified "
                             "dimensions, the following dimensions were "
                             "not found: %s" % repr(not_found))


    @classmethod
    def expanded(cls, arrays):
        return not any(array.shape not in [arrays[0].shape, (1,)] for array in arrays[1:])


    @classmethod
    def select_mask(cls, dataset, selection):
        """
        Given a Dataset object and a dictionary with dimension keys and
        selection keys (i.e tuple ranges, slices, sets, lists or literals)
        return a boolean mask over the rows in the Dataset object that
        have been selected.
        """
        mask = np.ones(len(dataset), dtype=np.bool)
        for dim, k in selection.items():
            if isinstance(k, tuple):
                k = slice(*k)
            arr = cls.values(dataset, dim)
            if isinstance(k, slice):
                if k.start is not None:
                    mask &= k.start <= arr
                if k.stop is not None:
                    mask &= arr < k.stop
            elif isinstance(k, (set, list)):
                iter_slcs = []
                for ik in k:
                    iter_slcs.append(arr == ik)
                mask &= np.logical_or.reduce(iter_slcs)
            elif callable(k):
                mask &= k(arr)
            else:
                index_mask = arr == k
                if dataset.ndims == 1 and np.sum(index_mask) == 0:
                    data_index = np.argmin(np.abs(arr - k))
                    mask = np.zeros(len(dataset), dtype=np.bool)
                    mask[data_index] = True
                else:
                    mask &= index_mask
        return mask


    @classmethod
    def indexed(cls, dataset, selection):
        """
        Given a Dataset object and selection to be applied returns
        boolean to indicate whether a scalar value has been indexed.
        """
        selected = list(selection.keys())
        all_scalar = all(not isinstance(sel, (tuple, slice, set, list))
                         for sel in selection.values())
        all_kdims = all(d in selected for d in dataset.kdims)
        return all_scalar and all_kdims and len(dataset.vdims) == 1


    @classmethod
    def range(cls, dataset, dimension):
        column = dataset.dimension_values(dimension)
        if dataset.get_dimension_type(dimension) is np.datetime64:
            return column.min(), column.max()
        else:
            try:
                return (np.nanmin(column), np.nanmax(column))
            except TypeError:
                column.sort()
                return column[0], column[-1]

    @classmethod
    def concatenate(cls, dataset, datatype=None):
        """
        Utility function to concatenate a list of Column objects,
        returning a new Dataset object. Note that this is unlike the
        .concat method which only concatenates the data.
        """
        if len(set(type(c) for c in dataset)) != 1:
               raise Exception("All inputs must be same type in order to concatenate")

        interfaces = set(c.interface for c in dataset)
        if len(interfaces)!=1 and datatype is None:
            raise Exception("Please specify the concatenated datatype")
        elif len(interfaces)!=1:
            interface = cls.interfaces[datatype]
        else:
            interface = interfaces.pop()

        concat_data = interface.concat(dataset)
        return dataset[0].clone(concat_data)

    @classmethod
    def reduce(cls, dataset, reduce_dims, function, **kwargs):
        kdims = [kdim for kdim in dataset.kdims if kdim not in reduce_dims]
        return cls.aggregate(dataset, kdims, function, **kwargs)

    @classmethod
    def array(cls, dataset, dimensions):
        return Element.array(dataset, dimensions)

    @classmethod
    def dframe(cls, dataset, dimensions):
        return Element.dframe(dataset, dimensions)

    @classmethod
    def columns(cls, dataset, dimensions):
        return Element.columns(dataset, dimensions)

    @classmethod
    def shape(cls, dataset):
        return dataset.data.shape

    @classmethod
    def length(cls, dataset):
        return len(dataset.data)

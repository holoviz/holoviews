"""
Data normalization operations.

Normalizing input data into a valid range is a common operation and
often required before further processing. The semantics of
normalization are dependent on the element type being normalized
making it difficult to provide a general and consistent interface.

The Normalization class is used to define such an interface and
subclasses are used to implement the appropriate normalization
operations per element type. Unlike display normalization, data
normalizations result in transformations to the stored data within
each element.
"""

import param
from ..core.operation import Operation
from ..element import Raster
from ..core import Overlay
from ..core.util import match_spec


class Normalization(Operation):
    """
    Base class for all normalization operation.

    This class standardizes how normalization is specified using the
    ranges and keys parameter. The ranges parameter is designed to be
    very flexible, allowing a concise description for simple
    normalization while allowing complex key- and element- specific
    normalization to also be specified.
    """

    data_range = param.Boolean(default=False, doc="""
       Whether normalization is allowed to use the minimum and maximum
       values of the existing data to infer an appropriate range""")

    ranges = param.ClassSelector(default={},  allow_None=True,
                                 class_=(dict, list), doc="""
       The simplest value of this parameter is None to skip all
       normalization. The next simplest value is an empty dictionary
       to only applies normalization to Dimensions with explicitly
       declared ranges.

       The next most common specification is a dictionary of values
       and tuple ranges. The value keys are the names of the
       dimensions to be normalized and the tuple ranges are of form
       (lower-bound, upper-bound). For instance, you could specify:

       {'Height':(0, 200), 'z':(0,1)}

       In this case, any element with a 'Height' or 'z'
       dimension (or both) will be normalized to the supplied ranges.

       Finally, element-specific normalization may also be specified
       by supplying a match tuple of form (<type>, <group>,
       <label>). A 1- or 2-tuple may be supplied by omitting the
       <group>, <label> or just the <label> components
       respectively. This tuple key then uses the dictionary
       value-range specification described above.

      For instance, you could normalize only the Image elements of
      group pattern using:

      {('Image','Pattern'):{'Height':(0, 200), 'z':(0,1)}})


      Key-wise normalization is possible for all these formats by
      supplying a list of such dictionary specification that will then
      be zipped with the keys parameter (if specified).
      """)

    keys = param.List(default=None, allow_None=True, doc="""
       If supplied, this list of keys is zipped with the supplied list
       of ranges.

       These keys are used to supply key specific normalization for
       HoloMaps containing matching key values, enabling per-element
       normalization.""")


    def __call__(self, element, ranges={}, keys=None, **params):
        params = dict(params,ranges=ranges, keys=keys)
        return super(Normalization, self).__call__(element, **params)


    def process_element(self, element, key, ranges={}, keys=None, **params):
        params = dict(params,ranges=ranges, keys=keys)
        self.p = param.ParamOverrides(self, params)
        return self._process(element, key)


    def get_ranges(self, element, key):
        """
        Method to get the appropriate normalization range dictionary
        given a key and element.
        """
        keys = self.p['keys']
        ranges = self.p['ranges']

        if ranges == {}:
            return {d.name: element.range(d.name, self.data_range)
                    for d in element.dimensions()}
        if keys is None:
            specs = ranges
        elif keys and not isinstance(ranges, list):
            raise ValueError("Key list specified but ranges parameter"
                             " not specified as a list.")
        elif len(keys) == len(ranges):
            # Unpack any 1-tuple keys
            try:
                index = keys.index(key)
                specs = ranges[index]
            except:
                raise KeyError("Could not match element key to defined keys")
        else:
            raise ValueError("Key list length must match length of supplied ranges")

        return match_spec(element, specs)


    def _process(self, view, key=None):
        raise NotImplementedError("Normalization not implemented")



class raster_normalization(Normalization):
    """
    Normalizes elements of type Raster.

    For Raster elements containing (NxM) data, this will normalize the
    array/matrix into the specified range if value_dimension matches
    a key in the ranges dictionary.

    For elements containing (NxMxD) data, the (NxM) components of the
    third dimensional are normalized independently if the
    corresponding value dimensions are selected by the ranges
    dictionary.
    """

    def _process(self, raster, key=None):
        if isinstance(raster, Raster):
            return self._normalize_raster(raster, key)
        elif isinstance(raster, Overlay):
            overlay_clone = raster.clone(shared_data=False)
            for k, el in raster.items():
                overlay_clone[k] =  self._normalize_raster(el, key)
            return overlay_clone
        else:
            raise ValueError("Input element must be a Raster or subclass of Raster.")


    def _normalize_raster(self, raster, key):
        if not isinstance(raster, Raster): return raster
        norm_raster = raster.clone(raster.data.copy())
        ranges = self.get_ranges(raster, key)

        for depth, name in enumerate(d.name for d in raster.vdims):
            depth_range = ranges.get(name, (None, None))
            if None in depth_range:  continue
            if depth_range and len(norm_raster.data.shape) == 2:
                depth_range = ranges[name]
                norm_raster.data[:,:] -= depth_range[0]
                range = (depth_range[1] - depth_range[0])
                if range:
                    norm_raster.data[:,:] /= range
            elif depth_range:
                norm_raster.data[:,:,depth] -= depth_range[0]
                range = (depth_range[1] - depth_range[0])
                if range:
                    norm_raster.data[:,:,depth] /= range
        return norm_raster



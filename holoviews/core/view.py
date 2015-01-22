"""
Supplies the View and Map abstract base classes. A View the basic data
structure that holds raw data and can be visualized. A Map is an
instance of NdMapping, a sliceable, multi-dimensional container that
holds View objects as values.
"""

import param
from .dimension import DimensionedData
from .options import options
from .ndmapping import NdMapping


class DataElement(DimensionedData):
    """
    A view is a data structure for holding data, which may be plotted
    using matplotlib. Views have an associated title and style
    name. All Views may be composed together into a GridLayout using
    the addition operator.
    """

    __abstract = True

    title = param.String(default='{label} {value}', doc="""
        The title formatting string allows the title to be composed from
        the view {label}, {value} quantity and view {type} but can also be set
        to a simple string.""")

    value = param.String(default='DataElement')

    options = options

    def __init__(self, data, **params):
        self._style = params.pop('style', None)
        super(DataElement, self).__init__(data, **params)




class UniformNdMapping(NdMapping):
    """
    A UniformNdMapping is a map of Views over a number of specified dimensions. The
    dimension may be a spatial dimension (i.e., a ZStack), time
    (specifying a frame sequence) or any other combination of
    Dimensions.  UniformNdMapping also adds handling of styles, appending the
    Dimension keys and values to titles and a number of methods to
    manipulate the Dimensions.

    UniformNdMapping objects can be sliced, sampled, reduced, overlaid and split
    along its and its containing Views dimensions. Subclasses should
    implement the appropriate slicing, sampling and reduction methods
    for their DataElement type.
    """

    title_suffix = param.String(default='\n {dims}', doc="""
       A string appended to the DataElement titles when they are added to the
       UniformNdMapping. Default adds a new line with the formatted dimensions
       of the UniformNdMapping inserted using the {dims} formatting keyword.""")

    value = param.String(default='UniformNdMapping')

    data_type = (DataElement, NdMapping)

    _abstract = True
    _deep_indexable = True
    _type = None
    _style = None

    @property
    def type(self):
        """
        The type of elements stored in the map.
        """
        if self._type is None:
            self._type = None if len(self) == 0 else self.last.__class__
        return self._type


    @property
    def style(self):
        """
        The style of elements stored in the map.
        """
        if self._style is None:
            self._style = None if len(self) == 0 else self.last.style
        return self._style


    @style.setter
    def style(self, style_name):
        self._style = style_name
        for val in self.values():
            val.style = style_name


    @property
    def empty_element(self):
        return self._type(None)


    def _item_check(self, dim_vals, data):
        if self.style is not None and self.style != data.style:
            data.style = self.style

        if self.type is not None and (type(data) != self.type):
            raise AssertionError("%s must only contain one type of DataElement." %
                                 self.__class__.__name__)
        super(UniformNdMapping, self)._item_check(dim_vals, data)


    def get_title(self, key, item, group_size=2):
        """
        Resolves the title string on the DataElement being added to the UniformNdMapping,
        adding the Maps title suffix.
        """
        if self.ndims == 1 and self.get_dimension('Default'):
            title_suffix = ''
        else:
            title_suffix = self.title_suffix
        dimension_labels = [dim.pprint_value(k) for dim, k in
                            zip(self.key_dimensions, key)]
        groups = [', '.join(dimension_labels[i*group_size:(i+1)*group_size])
                  for i in range(len(dimension_labels))]
        dims = '\n '.join(g for g in groups if g)
        title_suffix = title_suffix.format(dims=dims)
        return item.title + title_suffix


    def table(self, **kwargs):
        """
        Creates Table from all the elements in the UniformNdMapping.
        """

        table = None
        for key, value in self.data.items():
            value = value.table(**kwargs)
            for idx, (dim, val) in enumerate(zip(self.key_dimensions, key)):
                value = value.add_dimension(dim, idx, val)
            if table is None:
                table = value
            else:
                table.update(value)
        return table


__all__ = list(set([_k for _k, _v in locals().items()
                    if isinstance(_v, type) and issubclass(_v, DimensionedData)]))

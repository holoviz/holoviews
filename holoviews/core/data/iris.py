from __future__ import absolute_import

import datetime
from itertools import product

import iris
from iris.coords import DimCoord
from iris.cube import CubeList
from iris.experimental.equalise_cubes import equalise_attributes
from iris.util import guess_coord_axis

import numpy as np

from .interface import Interface, DataError
from .grid import GridInterface
from ..dimension import Dimension, asdim
from ..element import Element
from ..ndmapping import (NdMapping, item_check, sorted_context)
from ..spaces import HoloMap
from .. import util



def get_date_format(coord):
    def date_formatter(val, pos=None):
        date = coord.units.num2date(val)
        date_format = Dimension.type_formatters.get(datetime.datetime, None)
        if date_format:
            return date.strftime(date_format)
        else:
            return date

    return date_formatter


def coord_to_dimension(coord):
    """
    Converts an iris coordinate to a HoloViews dimension.
    """
    kwargs = {}
    if coord.units.is_time_reference():
        kwargs['value_format'] = get_date_format(coord)
    else:
        kwargs['unit'] = str(coord.units)
    return Dimension(coord.name(), **kwargs)


def sort_coords(coord):
    """
    Sorts a list of DimCoords trying to ensure that
    dates and pressure levels appear first and the
    longitude and latitude appear last in the correct
    order.
    """
    order = {'T': -2, 'Z': -1, 'X': 1, 'Y': 2}
    axis = guess_coord_axis(coord)
    return (order.get(axis, 0), coord and coord.name())



class CubeInterface(GridInterface):
    """
    The CubeInterface provides allows HoloViews to interact with iris
    Cube data. When passing an iris Cube to a HoloViews Element the
    init method will infer the dimensions of the Cube from its
    coordinates. Currently the interface only provides the basic
    methods required for HoloViews to work with an object.
    """

    types = (iris.cube.Cube,)

    datatype = 'cube'

    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        if kdims:
            kdims = [asdim(kd) for kd in kdims]
            kdim_names = [kd.name for kd in kdims]
        else:
            kdims = eltype.kdims
            kdim_names = [kd.name for kd in eltype.kdims]

        if not isinstance(data, iris.cube.Cube):
            if vdims is None:
                vdims = eltype.vdims
            ndims = len(kdim_names)
            vdim = asdim(vdims[0])
            vdims = [vdim]
            if isinstance(data, np.ndarray):
                if data.ndim != 2 or data.shape[1] != 2 or len(kdims) != 1:
                    raise ValueError('Iris interface could not interpret array data.')
                data = {kdims[0].name: data[:, 0], vdim.name: data[:, 1]}
            elif isinstance(data, tuple):
                value_array = data[-1]
                data = {d: vals for d, vals in zip(kdim_names + [vdim.name], data)}
            elif isinstance(data, list) and data == []:
                ndims = len(kdims)
                dimensions = [d.name for d in kdims+vdims]
                data = {d: np.array([]) for d in dimensions[:ndims]}
                data.update({d: np.empty((0,) * ndims) for d in dimensions[ndims:]})

            if isinstance(data, dict):
                value_array = data[vdim.name]
            coords = [(iris.coords.DimCoord(data[kd.name], long_name=kd.name,
                                            units=kd.unit), ndims-n-1)
                      for n, kd in enumerate(kdims)]
            try:
                data = iris.cube.Cube(value_array, long_name=vdim.name,
                                      dim_coords_and_dims=coords)
            except:
                pass
            if not isinstance(data, iris.cube.Cube):
                raise TypeError('Data must be be an iris Cube type.')

        if kdims:
            coords = []
            for kd in kdims:
                coord = data.coords(kd.name)
                if len(coord) == 0:
                    raise ValueError('Key dimension %s not found in '
                                     'Iris cube.' % kd)
                coords.append(kd if isinstance(kd, Dimension) else coord[0])
        else:
            coords = data.dim_coords
            coords = sorted(coords, key=sort_coords)
        kdims = [crd if isinstance(crd, Dimension) else coord_to_dimension(crd)
                 for crd in coords]
        if vdims is None:
            vdims = [Dimension(data.name(), unit=str(data.units))]

        return data, {'kdims':kdims, 'vdims':vdims}, {}


    @classmethod
    def validate(cls, dataset, vdims=True):
        if vdims and len(dataset.vdims) > 1:
            raise DataError("Iris cubes do not support more than one value dimension", cls)


    @classmethod
    def irregular(cls, dataset, dim):
        "CubeInterface does not support irregular data"
        return False


    @classmethod
    def shape(cls, dataset, gridded=False):
        if gridded:
            return dataset.data.shape
        else:
            return (cls.length(dataset), len(dataset.dimensions()))


    @classmethod
    def coords(cls, dataset, dim, ordered=False, expanded=False):
        dim = dataset.get_dimension(dim, strict=True)
        if expanded:
            return util.expand_grid_coords(dataset, dim.name)
        data = dataset.data.coords(dim.name)[0].points
        if ordered and np.all(data[1:] < data[:-1]):
            data = data[::-1]
        return data


    @classmethod
    def values(cls, dataset, dim, expanded=True, flat=True, compute=True):
        """
        Returns an array of the values along the supplied dimension.
        """
        dim = dataset.get_dimension(dim, strict=True)
        if dim in dataset.vdims:
            coord_names = [c.name() for c in dataset.data.dim_coords]
            data = dataset.data.copy().data
            data = cls.canonicalize(dataset, data, coord_names)
            return data.T.flatten() if flat else data
        elif expanded:
            data = cls.coords(dataset, dim.name, expanded=True)
            return data.T.flatten() if flat else data
        else:
            return cls.coords(dataset, dim.name, ordered=True)


    @classmethod
    def reindex(cls, dataset, kdims=None, vdims=None):
        dropped_kdims = [kd for kd in dataset.kdims if kd not in kdims]
        constant = {}
        for kd in dropped_kdims:
            vals = cls.values(dataset, kd.name, expanded=False)
            if len(vals) == 1:
                constant[kd.name] = vals[0]
        if len(constant) == len(dropped_kdims):
            constraints = iris.Constraint(**constant)
            return dataset.data.extract(constraints)
        elif dropped_kdims:
            return tuple(dataset.columns(kdims+vdims).values())
        return dataset.data


    @classmethod
    def groupby(cls, dataset, dims, container_type=HoloMap, group_type=None, **kwargs):
        """
        Groups the data by one or more dimensions returning a container
        indexed by the grouped dimensions containing slices of the
        cube wrapped in the group_type. This makes it very easy to
        break up a high-dimensional dataset into smaller viewable chunks.
        """
        if not isinstance(dims, list): dims = [dims]
        dims = [dataset.get_dimension(d, strict=True) for d in dims]
        constraints = [d.name for d in dims]
        slice_dims = [d for d in dataset.kdims if d not in dims]

        # Update the kwargs appropriately for Element group types
        group_kwargs = {}
        group_type = dict if group_type == 'raw' else group_type
        if issubclass(group_type, Element):
            group_kwargs.update(util.get_param_values(dataset))
            group_kwargs['kdims'] = slice_dims
        group_kwargs.update(kwargs)

        drop_dim = any(d not in group_kwargs['kdims'] for d in slice_dims)

        unique_coords = product(*[cls.values(dataset, d, expanded=False)
                                  for d in dims])
        data = []
        for key in unique_coords:
            constraint = iris.Constraint(**dict(zip(constraints, key)))
            extracted = dataset.data.extract(constraint)
            if drop_dim:
                extracted = group_type(extracted, kdims=slice_dims,
                                       vdims=dataset.vdims).columns()
            cube = group_type(extracted, **group_kwargs)
            data.append((key, cube))
        if issubclass(container_type, NdMapping):
            with item_check(False), sorted_context(False):
                return container_type(data, kdims=dims)
        else:
            return container_type(data)

    @classmethod
    def concat_dim(cls, datasets, dim, vdims):
        """
        Concatenates datasets along one dimension
        """
        cubes = []
        for c, cube in datasets.items():
            cube = cube.copy()
            cube.add_aux_coord(DimCoord([c], var_name=dim.name))
            cubes.append(cube)
        cubes = CubeList(cubes)
        equalise_attributes(cubes)
        return cubes.merge_cube()


    @classmethod
    def range(cls, dataset, dimension):
        """
        Computes the range along a particular dimension.
        """
        dim = dataset.get_dimension(dimension, strict=True)
        values = dataset.dimension_values(dim.name, False)
        return (np.nanmin(values), np.nanmax(values))


    @classmethod
    def redim(cls, dataset, dimensions):
        """
        Rename coords on the Cube.
        """
        new_dataset = dataset.data.copy()
        for name, new_dim in dimensions.items():
            if name == new_dataset.name():
                new_dataset.rename(new_dim.name)
            for coord in new_dataset.dim_coords:
                if name == coord.name():
                    coord.rename(new_dim.name)
        return new_dataset


    @classmethod
    def length(cls, dataset):
        """
        Returns the total number of samples in the dataset.
        """
        return np.product([len(d.points) for d in dataset.data.coords(dim_coords=True)], dtype=np.intp)


    @classmethod
    def sort(cls, columns, by=[], reverse=False):
        """
        Cubes are assumed to be sorted by default.
        """
        return columns


    @classmethod
    def aggregate(cls, columns, kdims, function, **kwargs):
        """
        Aggregation currently not implemented.
        """
        raise NotImplementedError


    @classmethod
    def sample(cls, dataset, samples=[]):
        """
        Sampling currently not implemented.
        """
        raise NotImplementedError


    @classmethod
    def add_dimension(cls, columns, dimension, dim_pos, values, vdim):
        """
        Adding value dimensions not currently supported by iris interface.
        Adding key dimensions not possible on dense interfaces.
        """
        if not vdim:
            raise Exception("Cannot add key dimension to a dense representation.")
        raise NotImplementedError


    @classmethod
    def select_to_constraint(cls, dataset, selection):
        """
        Transform a selection dictionary to an iris Constraint.
        """
        def get_slicer(start, end):
            def slicer(cell):
                return start <= cell.point < end
            return slicer
        constraint_kwargs = {}
        for dim, constraint in selection.items():
            if isinstance(constraint, slice):
                constraint = (constraint.start, constraint.stop)
            if isinstance(constraint, tuple):
                if constraint == (None, None):
                    continue
                constraint = get_slicer(*constraint)
            dim = dataset.get_dimension(dim, strict=True)
            constraint_kwargs[dim.name] = constraint
        return iris.Constraint(**constraint_kwargs)


    @classmethod
    def select(cls, dataset, selection_mask=None, **selection):
        """
        Apply a selection to the data.
        """
        constraint = cls.select_to_constraint(dataset, selection)
        pre_dim_coords = [c.name() for c in dataset.data.dim_coords]
        indexed = cls.indexed(dataset, selection)
        extracted = dataset.data.extract(constraint)
        if indexed and not extracted.dim_coords:
            return extracted.data.item()
        post_dim_coords = [c.name() for c in extracted.dim_coords]
        dropped = [c for c in pre_dim_coords if c not in post_dim_coords]
        for d in dropped:
            extracted = iris.util.new_axis(extracted, d)

        return extracted


Interface.register(CubeInterface)

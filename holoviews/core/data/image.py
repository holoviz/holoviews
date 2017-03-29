import numpy as np

from ..boundingregion import BoundingRegion, BoundingBox
from ..dimension import Dimension
from ..element import Element
from ..ndmapping import OrderedDict, NdMapping, item_check
from ..sheetcoords import SheetCoordinateSystem, Slice
from .. import util
from .grid import GridInterface
from .interface import Interface


class ImageInterface(GridInterface):
    """
    Interface for 2 or 3D arrays representing images
    of raw luminance values, RGB values or HSV values.
    """

    types = (np.ndarray,)

    datatype = 'image'

    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        if kdims is None:
            kdims = eltype.kdims
        if vdims is None:
            vdims = eltype.vdims

        kwargs = {}
        dimensions = [d.name if isinstance(d, Dimension) else
                      d for d in kdims + vdims]
        if isinstance(data, tuple):
            data = dict(zip(dimensions, data))
        if isinstance(data, dict):
            xs, ys = np.asarray(data[kdims[0].name]), np.asarray(data[kdims[1].name])
            l, r, xdensity, invertx = util.bound_range(xs, None)
            b, t, ydensity, inverty = util.bound_range(ys, None)
            kwargs['bounds'] = BoundingBox(points=((l, b), (r, t)))
            if len(vdims) == 1:
                data = np.flipud(np.asarray(data[vdims[0].name]))
            else:
                data = np.dstack([np.flipud(data[vd.name]) for vd in vdims])
            if invertx:
                data = data[:, ::-1]
            if inverty:
                data = data[::-1, :]
        if not isinstance(data, np.ndarray) or data.ndim not in [2, 3]:
            raise ValueError('ImageInterface expects a 2D array.')

        return data, {'kdims':kdims, 'vdims':vdims}, kwargs


    @classmethod
    def shape(cls, dataset):
        return cls.length(dataset), len(dataset.dimensions()),


    @classmethod
    def length(cls, dataset):
        print dataset.data.shape
        return np.product(dataset.data.shape)


    @classmethod
    def validate(cls, dataset):
        pass

    @classmethod
    def redim(cls, dataset, dimensions):
        return dataset.data

    @classmethod
    def reindex(cls, columns, kdims=None, vdims=None):
        data = columns.data
        if vdims is not None and vdims != columns.vdims and len(columns.vdims) > 1:
            inds = [columns.get_dimension_index(vd)-columns.ndims for vd in vdims]
            return data[:, :, inds] if len(inds) > 1 else data[:, :, inds[0]]
        return data

    @classmethod
    def range(cls, obj, dim):
        dim_idx = obj.get_dimension_index(dim)
        if dim_idx in [0, 1] and obj.bounds:
            l, b, r, t = obj.bounds.lbrt()
            if dim_idx:
                halfd = (1./obj.ydensity)/2.
                drange = (b+halfd, t-halfd)
            else:
                halfd = (1./obj.xdensity)/2.
                drange = (l+halfd, r-halfd)
        elif 1 < dim_idx < len(obj.vdims) + 2:
            dim_idx -= 2
            data = np.atleast_3d(obj.data)[:, :, dim_idx]
            drange = (np.nanmin(data), np.nanmax(data))
        else:
            drange = (None, None)
        return drange

    
    @classmethod
    def values(cls, dataset, dim, expanded=True, flat=True):
        """
        The set of samples available along a particular dimension.
        """
        dim_idx = dataset.get_dimension_index(dim)
        if dim_idx in [0, 1]:
            l, b, r, t = dataset.bounds.lbrt()
            dim2, dim1 = dataset.data.shape[:2]
            d1_half_unit = float(r - l)/dim1/2.
            d2_half_unit = float(t - b)/dim2/2.
            d1lin = np.linspace(l+d1_half_unit, r-d1_half_unit, dim1)
            d2lin = np.linspace(b+d2_half_unit, t-d2_half_unit, dim2)
            if expanded:
                values = np.meshgrid(d2lin, d1lin)[abs(dim_idx-1)]
                return values.flatten() if flat else values
            else:
                return d2lin if dim_idx else d1lin
        elif dataset.ndims <= dim_idx < len(dataset.dimensions()):
            # Raster arrays are stored with different orientation
            # than expanded column format, reorient before expanding
            if dataset.data.ndim > 2:
                data = dataset.data[:, :, dim_idx-dataset.ndims]
            else:
                data = dataset.data
            data = np.flipud(data)
            return data.T.flatten() if flat else data
        else:
            return None


    @classmethod
    def select(cls, dataset, selection_mask=None, **selection):
        """
        Slice the underlying numpy array in sheet coordinates.
        """
        selection = {k: slice(*sel) if isinstance(sel, tuple) else sel
                     for k, sel in selection.items()}
        coords = tuple(selection[kd.name] if kd.name in selection else slice(None)
                       for kd in dataset.kdims)
        if not any([isinstance(el, slice) for el in coords]):
            return dataset.data[dataset.sheet2matrixidx(*coords)]

        # Compute new bounds
        ys, xs = dataset.data.shape[:2]
        xidx, yidx = coords
        l, b, r, t = dataset.bounds.lbrt()
        xdensity, ydensity = dataset.xdensity, dataset.ydensity
        xunit = (1./xdensity)
        yunit = (1./ydensity)
        if isinstance(xidx, slice):
            l = l if xidx.start is None else max(l, xidx.start)
            r = r if xidx.stop is None else min(r, xidx.stop)
        if isinstance(yidx, slice):
            b = b if yidx.start is None else max(b, yidx.start)
            t = t if yidx.stop is None else min(t, yidx.stop)
        bounds = BoundingBox(points=((l, b), (r, t)))

        # Apply new bounds
        slc = Slice(bounds, dataset)
        data = slc.submatrix(dataset.data)

        # Apply scalar and list indices
        l, b, r, t = slc.compute_bounds(dataset).lbrt()
        if not isinstance(xidx, slice):
            if not isinstance(xidx, (list, set)): xidx = [xidx]
            if len(xidx) > 1:
                xdensity = xdensity*(float(len(xidx))/xs)
            idxs = []
            ls, rs = [], []
            for idx in xidx:
                xc, _ = dataset.closest_cell_center(idx, b)
                ls.append(xc-xunit/2)
                rs.append(xc+xunit/2)
                _, x = dataset.sheet2matrixidx(idx, b)
                idxs.append(x)
            l, r = np.min(ls), np.max(rs)
            data = data[:, np.array(idxs)]
        elif not isinstance(yidx, slice):
            if not isinstance(yidx, (set, list)): yidx = [yidx]
            if len(yidx) > 1:
                ydensity = ydensity*(float(len(yidx))/ys)
            idxs = []
            bs, ts = [], []
            for idx in yidx:
                _, yc = dataset.closest_cell_center(l, idx)
                bs.append(yc-yunit/2)
                ts.append(yc+yunit/2)
                y, _ = dataset.sheet2matrixidx(l, idx)
                idxs.append(y)
            b, t = np.min(bs), np.max(ts)
            data = data[np.array(idxs), :]
        return data


    @classmethod
    def sample(cls, dataset, samples=[]):
        """
        Sample the Raster along one or both of its dimensions,
        returning a reduced dimensionality type, which is either
        a ItemTable, Curve or Scatter. If two dimension samples
        and a new_xaxis is provided the sample will be the value
        of the sampled unit indexed by the value in the new_xaxis
        tuple.
        """
        if len(samples[0]) == 1:
            return dataset.select(**{dataset.kdims[0].name:
                                     [s[0] for s in samples]}).columns()
        else:
            return [c+(dataset.data[dataset._coord2matrix(c)],)
                    for c in samples]

    @classmethod
    def length(cls, dataset):
        return np.product(dataset.data.shape)


    @classmethod
    def groupby(cls, dataset, dim_names, container_type, group_type, **kwargs):
        # Get dimensions information
        dimensions = [dataset.get_dimension(d) for d in dim_names]
        kdims = [kdim for kdim in dataset.kdims if kdim not in dimensions]

        # Update the kwargs appropriately for Element group types
        group_kwargs = {}
        group_type = dict if group_type == 'raw' else group_type
        if issubclass(group_type, Element):
            group_kwargs.update(util.get_param_values(dataset))
            group_kwargs['kdims'] = kdims
        group_kwargs.update(kwargs)

        if len(dimensions) == 1:
            didx = dataset.get_dimension_index(dimensions[0])
            coords = dataset.dimension_values(dimensions[0], False)
            xvals = dataset.dimension_values(abs(didx-1), False)
            samples = [(i, slice(None)) if didx else (slice(None), i)
                       for i in range(dataset.data.shape[abs(didx-1)])]
            if didx:
                samples = samples[::-1]
                data = dataset.data
            else:
                data = dataset.data[::-1, :]
            groups = [(c, group_type((xvals, data[s]), **group_kwargs))
                       for s, c in zip(samples, coords)]
        else:
            data = zip(*[dataset.dimension_values(i) for i in range(len(dataset.dimensions()))])
            groups = [(g[:dataset.ndims], group_type([g[dataset.ndims:]], **group_kwargs))
                      for g in data]

        if issubclass(container_type, NdMapping):
            with item_check(False):
                return container_type(groups, kdims=dimensions)
        else:
            return container_type(grouped_data)


    @classmethod
    def unpack_scalar(cls, dataset, data):
        """
        Given a dataset object and data in the appropriate format for
        the interface, return a simple scalar.
        """
        if np.isscalar(data) or len(data) != 1:
            return data
        key = list(data.keys())[0]

        if len(data[key]) == 1 and key in dataset.vdims:
            return data[key][0]


    @classmethod
    def aggregate(cls, dataset, kdims, function, **kwargs):
        kdims = [kd.name if isinstance(kd, Dimension) else kd for kd in kdims]
        axes = tuple(dataset.ndims-dataset.get_dimension_index(kdim)-1
                     for kdim in dataset.kdims if kdim not in kdims)

        data = np.atleast_1d(function(dataset.data, axis=axes, **kwargs))
        if not kdims and len(dataset.vdims) == 1:
            if np.isscalar(data):
                return data
            else:
                return data[0]
        elif len(axes) == 1:
            return {kdims[0]: cls.values(dataset, axes[0], expanded=False),
                    dataset.vdims[0].name: data[::-1] if axes[0] else data}


Interface.register(ImageInterface)

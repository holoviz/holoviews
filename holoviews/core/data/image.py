import numpy as np

from ..boundingregion import BoundingRegion, BoundingBox
from ..dimension import Dimension
from ..ndmapping import OrderedDict
from ..sheetcoords import SheetCoordinateSystem, Slice
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

        dimensions = [d.name if isinstance(d, Dimension) else
                      d for d in kdims + vdims]
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError('ImageIntereface expects a 2D array.')

        if kdims is None:
            kdims = eltype.kdims
        if vdims is None:
            vdims = eltype.vdims
        return data, {'kdims':kdims, 'vdims':vdims}, {}


    @classmethod
    def validate(cls, dataset):
        pass


    @classmethod
    def range(cls, obj, dim):
        dim_idx = obj.get_dimension_index(dim)
        if dim_idx in [0, 1] and obj.bounds:
            l, b, r, t = obj.bounds.lbrt()
            if dim_idx:
                drange = (b, t)
            else:
                drange = (l, r)
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
            d1_half_unit = (r - l)/dim1/2.
            d2_half_unit = (t - b)/dim2/2.
            d1lin = np.linspace(l+d1_half_unit, r-d1_half_unit, dim1)
            d2lin = np.linspace(b+d2_half_unit, t-d2_half_unit, dim2)
            if expanded:
                values = np.meshgrid(d2lin, d1lin)[abs(dim_idx-1)]
                return values.flatten() if flat else values
            else:
                return d2lin if dim_idx else d1lin
        elif dim_idx == 2:
            # Raster arrays are stored with different orientation
            # than expanded column format, reorient before expanding
            data = np.flipud(dataset.data)
            return data.flatten() if flat else data
        else:
            return None, None


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
            data = dataset.data[dataset.sheet2matrixidx(*coords)]
        xidx, yidx = coords
        l, b, r, t = dataset.bounds.lbrt()
        xunit = (1./dataset.xdensity)
        yunit = (1./dataset.ydensity)    
        if isinstance(xidx, slice):
            l = l if xidx.start is None else max(l, xidx.start)
            r = r if xidx.stop is None else min(r, xidx.stop)
        if isinstance(yidx, slice):
            b = b if yidx.start is None else max(b, yidx.start)
            t = t if yidx.stop is None else min(t, yidx.stop)
        bounds = BoundingBox(points=((l, b), (r, t)))
        slc = Slice(bounds, dataset)
        data = slc.submatrix(dataset.data)
        l, b, r, t = slc.compute_bounds(dataset).lbrt()
        if not isinstance(xidx, slice):
            xc, _ = dataset.closest_cell_center(xidx, b)
            l, r = xc-xunit/2, xc+xunit/2
            _, x = dataset.sheet2matrixidx(xidx, b)
            data = data[:, x][:, np.newaxis]
        elif not isinstance(yidx, slice):
            _, yc = dataset.closest_cell_center(l, yidx)
            b, t = yc-yunit/2, yc+yunit/2
            y, _ = dataset.sheet2matrixidx(l, yidx)
            data = data[y, :][np.newaxis, :]
        bounds = BoundingBox(points=((l, b), (r, t)))
        return data, {'bounds': bounds}


    @classmethod
    def length(cls, dataset):
        return np.product(dataset.data.shape)
    

    @classmethod
    def aggregate(cls, dataset, kdims, function, **kwargs):
        kdims = [kd.name if isinstance(kd, Dimension) else kd for kd in kdims]
        axes = tuple(dataset.ndims-dataset.get_dimension_index(kdim)-1
                     for kdim in dataset.kdims if kdim not in kdims)
        
        data = np.atleast_1d(function(dataset.data, axis=axes, **kwargs))
        if np.isscalar(data):
            return data
        elif len(axes) == 1:
            return {kdims[0]: cls.values(dataset, axes[0], expanded=False),
                    dataset.vdims[0].name: data}


Interface.register(ImageInterface)

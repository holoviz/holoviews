import numpy as np

import param

from ..core import Dimension, Dataset, Element2D


class Geometry(Dataset, Element2D):
    """
    Geometry elements represent a collection of objects drawn in
    a 2D coordinate system. The two key dimensions correspond to the
    x- and y-coordinates in the 2D space, while the value dimensions
    may be used to control other visual attributes of the Geometry
    """

    group = param.String(default='Geometry', constant=True)

    kdims = param.List(default=[Dimension('x'), Dimension('y')],
                       bounds=(2, 2), constant=True, doc="""
        The key dimensions of a geometry represent the x- and y-
        coordinates in a 2D space.""")

    vdims = param.List(default=[], constant=True, doc="""
        Value dimensions can be associated with a geometry.""")

    __abstract = True


class Points(Geometry):
    """
    Points represents a set of coordinates in 2D space, which may
    optionally be associated with any number of value dimensions.
    """

    group = param.String(default='Points', constant=True)

    _auto_indexable_1d = True


class VectorField(Geometry):
    """
    A VectorField represents a set of vectors in 2D spac with an
    associated angle, as well as an optional magnitude and any number
    of other value dimensions. The angles are assumed to be defined in
    radians and by default the magnitude is assumed to be normalized
    to be between 0 and 1.
    """

    group = param.String(default='VectorField', constant=True)

    vdims = param.List(default=[Dimension('Angle', cyclic=True, range=(0,2*np.pi)),
                                Dimension('Magnitude')], bounds=(1, None))

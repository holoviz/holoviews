"""
Provides SheetCoordinateSystem, allowing conversion between the two
coordinate systems used in ImaGen.

'Sheet coordinates' allow simulation parameters to be specified in
units that are density-independent, whereas 'matrix coordinates'
provide a means of realizing the continuous sheets.

Hence we can have a pair of 'sheet coordinates' (x,y); floating-point
Cartesian coordinates indicating an arbitrary point on the sheet's plane.
We can also have a pair of 'matrix coordinates' (r,c), which are used
to address an underlying matrix. These matrix coordinates are also
floating-point numbers to allow precise conversion between the two
schemes. Where it is necessary to address a specific element of the
matrix (as is often the case in calculations), we also have the usual
matrix index coordinates (r_idx, c_idx). We refer to these as
matrixidx coordinates. SheetCoordinateSystem provies methods for converting
between sheet and matrix coordinates, as well as sheet and matrixidx
coordinates.

Everyone should use these facilities for conversions between the two
coordinate systems to guarantee consistency.




Example of how the matrix stores the representation of the Sheet
================================================================

For the purposes of this example, assume the goal is a Sheet with
density=3 that has a 1 at (-1/2,-1/2), a 5 at (0.0,0.0), and a 9 at
(1/2,1/2).  More precisely, for this Sheet,

the continuous area from -1/2,-1/2 to -1/6,-1/6 has value 1,
the continuous area from -1/6,-1/6 to  1/6,1/6  has value 5, and
the continuous area from  1/6,1/6  to  1/2,1/2  has value 9.

With the rest of the elements filled in, the Sheet would look like::

    (-1/2,1/2) -+-----+-----+-----+- (1/2,1/2)
                |     |     |     |
                |  7  |  8  |  9  |
                |     |     |     |
    (-1/2,1/6) -+-----+-----+-----+- (1/2,1/6)
                |     |     |     |
                |  4  |  5  |  6  |
                |     |     |     |
   (-1/2,-1/6) -+-----+-----+-----+- (1/2,-1/6)
                |     |     |     |
                |  1  |  2  |  3  |
                |     |     |     |
   (-1/2,-1/2) -+-----+-----+-----+- (1/2,-1/2)

where element 5 is centered on 0.0,0.0.  A matrix that would match
these Sheet coordinates is::

  [[7 8 9]
   [4 5 6]
   [1 2 3]]

If we have such a matrix, we can access it in one of two ways: Sheet
or matrix/matrixidx coordinates.  In matrixidx coordinates, the matrix is indexed
by rows and columns, and it is possible to ask for the element at
location [0,2] (which returns 9 as in any normal row-major matrix).
But the values can additionally be accessed in Sheet coordinates,
where the matrix is indexed by a point in the Cartesian plane.  In
Sheet coordinates, it is possible to ask for the element at location
(0.3,0.02), which returns floating-point matrix coordinates that can
be cropped to give the nearest matrix element, namely the one with
value 6.

Of course, it would be an error to try to pass matrix coordinates like
[0,2] to the sheet2matrix calls; the result would be a value far
outside of the actual matrix.
"""

__version__ = '$Revision$'


from numpy import array,floor,ceil,round_,arange

from boundingregion import BoundingBox





# Note about the 'bounds-master' approach we have adopted
# =======================================================
#
# Our current approach is a "bounds-master" approach, where we trust
# the user's specified x width, and choose the nearest density and y
# height to make that possible.  The advantage of this is that when we
# change the density (which is often), each such simulation is the
# best possible approximation to the given area.  Generally, the area
# is much more meaningful than the density, so this approach makes
# sense.  Plus, the y height is usually the same as the x width, so
# it's not usually a problem that the y height is not respected.  The
# disadvantage is that the user's area can only be trusted in one
# dimension, because of wanting to avoid the complication of separate
# xdensity and ydensity, which makes this approach very difficult to
# explain to the user.
#
# The other approach is density-master: trust the user's specified
# density as-is, and simply choose the nearest area that fits that
# density.  The advantages are that (a) it's very simple to describe
# and reason about, and (b) if a user asks for a different area, they
# get a true subsection of the same simulation they would have gotten
# at the larger area.  The disadvantage is that the simulation isn't
# the best approximation of the given area that it could be -- e.g. at
# low densities, the sheet area could be quite significantly different
# than the one the user requested.  Plus, if we took this approach
# seriously, then we'd let the density specify the matrix coordinate
# system entirely, including the origin, which would mean that the
# actual area would often be offset from the intended one, which is
# even worse.  Differences between the area and the offset could cause
# severe problems in the alignment of projections between sheets with
# different densities, which would make low-density versions of
# hierarchical models behave very strangely.




class SheetCoordinateSystem(object):
    """
    Provides methods to allow conversion between sheet and matrix
    coordinates.
    """
    def __get_xdensity(self):
        return self.__xdensity
    def __get_ydensity(self):
        return self.__ydensity
    def __get_shape(self):
        return self.__shape

    #### These are all properties so that they can't be set ####
    # CB: unnecessary? if we're going to keep this, what about bounds and lbrt?
    xdensity = property(__get_xdensity,
                        doc="""The spacing between elements in an underlying
                        matrix representation, in the x direction.""")

    ydensity = property(__get_ydensity,
                        doc="""The spacing between elements in an underlying
                        matrix representation, in the y direction.""")

    shape = property(__get_shape)


    def __init__(self,bounds,xdensity,ydensity=None):
        """
        Store the bounds (as l,b,r,t in an array), xdensity, and
        ydensity.

        If ydensity is not specified, it is assumed that the specified
        xdensity is nominal and that the true xdensity should be
        calculated. The top and bottom bounds are adjusted so that
        the ydensity is equal to the xdensity.

        If both xdensity and ydensity are specified, these and the bounds
        are taken to be exact and are not adjusted.
        """
        if not ydensity:
            bounds,xdensity = self.__equalize_densities(bounds,xdensity)

        self.bounds = bounds
        self.__set_xdensity(xdensity)
        self.__set_ydensity(ydensity or xdensity)

        self.lbrt = array(bounds.lbrt())

        r1,r2,c1,c2 = Slice._boundsspec2slicespec(self.lbrt,self)
        self.__shape = (r2-r1,c2-c1)


    ### we use xstep and ystep so that the repeatedly performed
    ### calculations in matrix2sheet() use multiplications rather than
    ### divisions, for speed
    def __set_xdensity(self,density):
        self.__xdensity=density
        self.__xstep = 1.0/density

    def __set_ydensity(self,density):
        self.__ydensity=density
        self.__ystep = 1.0/density


    def __equalize_densities(self,nominal_bounds,nominal_density):
        """
        Calculate the true density along x, and adjust the top and
        bottom bounds so that the density along y will be equal.

        Returns (adjusted_bounds,true_density)
        """
        left,bottom,right,top = nominal_bounds.lbrt()
        width = right-left; height = top-bottom
        center_y = bottom + height/2.0
        # The true density is not equal to the nominal_density
        # when nominal_density*(right-left) is not an integer.
        true_density = int(nominal_density*(width))/float(width)

        n_cells = round(height*true_density,0)
        adjusted_half_height = n_cells/true_density/2.0
        # (The above might be clearer as (step*n_units)/2.0, where
        # step=1.0/density.)

        return (BoundingBox(points=((left,  center_y-adjusted_half_height),
                                    (right, center_y+adjusted_half_height))),
                true_density)


    def sheet2matrix(self,x,y):
        """
        Convert a point (x,y) in Sheet coordinates to continuous matrix
        coordinates.

        Returns (float_row,float_col), where float_row corresponds to y,
        and float_col to x.

        Valid for scalar or array x and y.

        Note about Bounds
        For a Sheet with BoundingBox(points=((-0.5,-0.5),(0.5,0.5)))
        and density=3, x=-0.5 corresponds to float_col=0.0 and x=0.5
        corresponds to float_col=3.0.  float_col=3.0 is not inside the
        matrix representing this Sheet, which has the three columns
        (0,1,2). That is, x=-0.5 is inside the BoundingBox but x=0.5
        is outside. Similarly, y=0.5 is inside (at row 0) but y=-0.5
        is outside (at row 3) (it's the other way round for y because
        the matrix row index increases as y decreases).
        """
        # First translate to (left,top), which is [0,0] in the matrix,
        # then scale to the size of the matrix. The y coordinate needs
        # to be flipped, because the points are moving down in the
        # sheet as the y index increases in the matrix.
        float_col = (x-self.lbrt[0]) * self.__xdensity
        float_row = (self.lbrt[3]-y) * self.__ydensity
        return float_row, float_col


    def sheet2matrixidx(self,x,y):
        """
        Convert a point (x,y) in sheet coordinates to the integer row and
        column index of the matrix cell in which that point falls, given a
        bounds and density.  Returns (row,column).

        Note that if coordinates along the right or bottom boundary are
        passed into this function, the returned matrix coordinate of the
        boundary will be just outside the matrix, because the right and
        bottom boundaries are exclusive.

        Valid for scalar or array x and y.
        """
        r,c = self.sheet2matrix(x,y)
        r = floor(r)
        c = floor(c)

        # CB: was it better to have two different methods?
        if hasattr(r,'astype'):
            return r.astype(int), c.astype(int)
        else:
            return int(r),int(c)


    def matrix2sheet(self,float_row,float_col):
        """
        Convert a floating-point location (float_row,float_col) in matrix
        coordinates to its corresponding location (x,y) in sheet
        coordinates.

        Valid for scalar or array float_row and float_col.

        Inverse of sheet2matrix().
        """
        x = float_col*self.__xstep + self.lbrt[0]
        y = self.lbrt[3] - float_row*self.__ystep
        return x, y


    def matrixidx2sheet(self,row,col):
        """
        Return (x,y) where x and y are the floating point coordinates of
        the *center* of the given matrix cell (row,col). If the matrix cell
        represents a 0.2 by 0.2 region, then the center location returned
        would be 0.1,0.1.

        NOTE: This is NOT the strict mathematical inverse of
        sheet2matrixidx(), because sheet2matrixidx() discards all but the integer
        portion of the continuous matrix coordinate.

        Valid only for scalar or array row and col.
        """
        x,y = self.matrix2sheet((row+0.5), (col+0.5))

        # Rounding is useful for comparing the result with a floating point number
        # that we specify by typing the number out (e.g. fp = 0.5).
        # Round eliminates any precision errors that have been compounded
        # via floating point operations so that the rounded number will better
        # match the floating number that we type in.
        return round_(x,10),round_(y,10)


    def closest_cell_center(self,x,y):
        """
        Given arbitary sheet coordinates, return the sheet coordinates
        of the center of the closest unit.
        """
        return self.matrixidx2sheet(*self.sheet2matrixidx(x,y))


    def sheetcoordinates_of_matrixidx(self):
        """
        Return x,y where x is a vector of sheet coordinates
        representing the x-center of each matrix cell, and y
        represents the corresponding y-center of the cell.
        """
        rows,cols = self.shape
        return self.matrixidx2sheet(arange(rows),arange(cols))







# Needs cleanup/rename:
#
# since it's different from a Python slice.  It's our special slice
# that's an array specifying row_start,row_stop,col_start,col_stop for
# a Sheet (2d array).
#
# In python, a[slice(0,2)] (where a is a list/array/similar) is
# equivalent to a[0:2].
#
# So, not sure what to call this class. (Will need to rename some
# methods, too.) SCSSlice? SheetSlice?

from numpy import int32,ndarray

class Slice(ndarray):
    """
    Represents a slice of a SheetCoordinateSystem; i.e., an array
    specifying the row and column start and end points for a submatrix
    of the SheetCoordinateSystem.

    The slice is created from the supplied bounds by calculating the
    slice that corresponds most closely to the specified bounds.
    Therefore, the slice does not necessarily correspond exactly to
    the specified bounds. The bounds that do exactly correspond to the
    slice are available via the 'bounds' attribute.

    Note that the slice does not respect the bounds of the
    SheetCoordinateSystem, and that actions such as translate() also
    do not respect the bounds. To ensure that the slice is within the
    SheetCoordinateSystem's bounds, use crop_to_sheet().
    """

    def compute_bounds(self,scs):
        spec = self._slicespec2boundsspec(self,scs)
        return BoundingBox(points=spec)

    __slots__ = []

    def __new__(cls, bounds, sheet_coordinate_system, force_odd=False,
                min_matrix_radius=1): # CEBALERT: min_matrix_radius only used for odd slice.
        """
        Create a slice of the given sheet_coordinate_system from the
        specified bounds.
        """
        # I couldn't find documentation on subclassing array; I used
        # the following as reference:
        # http://scipy.org/scipy/numpy/browser/branches/maskedarray/
        #  numpy/ma/tests/test_subclassing.py?rev=4577
        if force_odd:
            slicespec=Slice._createoddslicespec(bounds,sheet_coordinate_system,
                                                min_matrix_radius)
        else:
            slicespec=Slice._boundsspec2slicespec(bounds.lbrt(),sheet_coordinate_system)
        # numpy.int32 is specified explicitly in Slice to avoid having
        # it default to numpy.int. int32 saves memory (and is expected
        # by optimized C functions).
        a = array(slicespec, dtype=int32, copy=False).view(cls)
        return a


    def submatrix(self,matrix):
        """
        Return the submatrix of the given matrix specified by this
        slice.

        Equivalent to computing the intersection between the
        SheetCoordinateSystem's bounds and the bounds, and
        returning the corresponding submatrix of the given matrix.

        The submatrix is just a view into the sheet_matrix; it is not
        an independent copy.
        """
        return matrix[self[0]:self[1],self[2]:self[3]]

    # CB: not sure if this is a good idea or not. In some ways, I
    # think matrix[slice_()] would be clearer than
    # slice.submatrix(matrix), but I'm not sure. cf.py is where
    # to look to see this in action.
##     def __call__(self):
##         return slice(self[0],self[1]),slice(self[2],self[3])

    ### CLEANUP ###

    # CEBALERT: unnecessary? use translate and crop and back. or is
    # that more steps? rename
    def positionlesscrop(self,x,y,sheet_coord_system):
        """
        Return the correct slice for a weights/mask matrix at this
        ConnectionField's location on the sheet (i.e. for getting
        the correct submatrix of the weights or mask in case the
        unit is near the edge of the sheet).
        """
        sheet_rows,sheet_cols = sheet_coord_system.shape

        # get size of weights matrix
        n_rows,n_cols = self.shape_on_sheet()

        # get slice for the submatrix
        center_row,center_col = sheet_coord_system.sheet2matrixidx(x,y)

        c1 = -min(0, center_col-n_cols/2)  # assume odd weight matrix so can use n_cols/2
        r1 = -min(0, center_row-n_rows/2)  # for top and bottom
        c2 = -max(-n_cols, center_col-sheet_cols-n_cols/2)
        r2 = -max(-n_rows, center_row-sheet_rows-n_rows/2)

        self.set((r1,r2,c1,c2))

    # CBALERT: assumes the user wants the bounds to be centered about
    # the unit, which might not be true.
    def positionedcrop(self,x,y,sheet_coord_system):
        """
        Offset the bounds_template to this cf's location and store the
        result in the 'bounds' attribute.

        Also stores the input_sheet_slice for access by C.
        """
        # translate to this cf's location
        cf_row,cf_col = sheet_coord_system.sheet2matrixidx(x,y)

        # should result in same no. of comps of right bounds as before during init,
        # since i removed one calc from init

        bounds_x,bounds_y=self.compute_bounds(sheet_coord_system).centroid()

        b_row,b_col=sheet_coord_system.sheet2matrixidx(bounds_x,bounds_y)

        row_offset = cf_row-b_row
        col_offset = cf_col-b_col
        self.translate(row_offset,col_offset)

    ###############


    def translate(self, r, c):
        """
        Translate the slice by the specified number of rows
        and columns.
        """
        self+=[r,r,c,c]

    def set(self,slice_specification):
        """Set this slice from some iterable that specifies (r1,r2,c1,c2)."""
        self.put([0,1,2,3],slice_specification) # pylint: disable-msg=E1101

    def shape_on_sheet(self):
        """Return the shape of the array that this Slice would give on its sheet."""
        return self[1]-self[0],self[3]-self[2]

    def crop_to_sheet(self,sheet_coord_system):
        """Crop the slice to the SheetCoordinateSystem's bounds."""
        maxrow,maxcol = sheet_coord_system.shape

        self[0] = max(0,self[0])
        self[1] = min(maxrow,self[1])
        self[2] = max(0,self[2])
        self[3] = min(maxcol,self[3])


    ### CB: working on methods below here
    # (+ not keeping these names)


    @staticmethod
    def _createoddslicespec(bounds,scs,min_matrix_radius):
        """
        Create the 'odd' Slice that best approximates the specifed
        sheet-coordinate bounds.

        The supplied bounds are translated to have a center at the
        center of one of the sheet's units (we arbitrarily use the
        center unit), and then these bounds are converted to a slice
        in such a way that the slice exactly includes all units whose
        centers are within the bounds (see boundsspec2slicespec()).
        However, to ensure that the bounds are treated symmetrically,
        we take the right and bottom bounds and reflect these about
        the center of the slice (i.e. we take the 'xradius' to be
        right_col-center_col and the 'yradius' to be
        bottom_col-center_row). Hence, if the bounds happen to go
        through units, if the units are included on the right and
        bottom bounds, they will be included on the left and top
        bounds. This ensures that the slice has odd dimensions.
        """
        bounds_xcenter,bounds_ycenter=bounds.centroid()
        sheet_rows,sheet_cols = scs.shape

        # arbitrary (e.g. could use 0,0)
        center_row,center_col = sheet_rows/2,sheet_cols/2
        unit_xcenter,unit_ycenter=scs.matrixidx2sheet(center_row,
                                                      center_col)

        bounds.translate(unit_xcenter-bounds_xcenter,
                         unit_ycenter-bounds_ycenter)

        ########## CEBALERT: assumes weights are to be centered about each unit.
        r1,r2,c1,c2 = Slice._boundsspec2slicespec(bounds.lbrt(),scs)

        # use the calculated radius unless it's smaller than the min
        xrad=max(c2-center_col-1,min_matrix_radius)
        yrad=max(r2-center_row-1,min_matrix_radius)

        r2=center_row+yrad+1
        c2=center_col+xrad+1
        r1=center_row-yrad
        c1=center_col-xrad
        ##########

        # weights matrix must be odd (otherwise this method has an error)
        # CEBALERT: this test should move to a test file.
##         if rows%2!=1 or cols%2!=1:
##             raise AssertionError("nominal_bounds_template yielded even-height or even-width weights matrix (%s rows, %s columns) - weights matrix must have odd dimensions."%(rows,cols))

        return (r1,r2,c1,c2)


##         # CEBALERT: with min_matrix_radius, this test is unnecessary? (check)
##         # user-supplied bounds must lead to a weights matrix of at least 1x1
##         rows,cols = weights_slice.shape_on_sheet()
##         if rows==0 or cols==0:
##             raise ValueError("nominal_bounds_template results in a zero-sized weights matrix (%s,%s) - you may need to supply a larger nominal_bounds_template or increase the density of the sheet."%(rows,cols))




    @staticmethod
    def _boundsspec2slicespec(boundsspec,scs):
        """
        Convert an iterable boundsspec (supplying l,b,r,t of a
        BoundingRegion) into a Slice specification.

        Includes all units whose centers are within the specified
        sheet-coordinate bounds specified by boundsspec.

        Exact inverse of _slicespec2boundsspec().
        """
        l,b,r,t = boundsspec

        t_m,l_m = scs.sheet2matrix(l,t)
        b_m,r_m = scs.sheet2matrix(r,b)

        l_idx = int(ceil(l_m-0.5))
        t_idx = int(ceil(t_m-0.5))
        # CBENHANCEMENT: Python 2.6's math.trunc()?
        r_idx = int(floor(r_m+0.5))
        b_idx = int(floor(b_m+0.5))

        return t_idx,b_idx,l_idx,r_idx


    @staticmethod
    def _slicespec2boundsspec(slicespec,scs):
        """
        Convert an iterable slicespec (supplying r1,r2,c1,c2 of a
        Slice) into a BoundingRegion specification.

        Exact inverse of _boundsspec2slicespec().
        """
        r1,r2,c1,c2 = slicespec

        left,bottom = scs.matrix2sheet(r2,c1)
        right, top  = scs.matrix2sheet(r1,c2)

        return ((left,bottom),(right,top))

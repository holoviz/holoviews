.. holoviews documentation master file, created by
   sphinx-quickstart on Wed May 14 14:25:57 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. notebook:: imagen index.ipynb

..
   # Code used to generate mandlebrot.npy
   from numpy import *
   import pylab

   def mandelbrot( h,w, maxit=200 ):
           y,x = ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j ]
           c = x+y*1j
           z = c
           divtime = maxit + zeros(z.shape, dtype=int)
           for i in xrange(maxit):
                   z  = z**2 + c
                   diverge = z*conj(z) > 2**2
                   div_now = diverge & (divtime==maxit)
                   divtime[div_now] = i
                   z[diverge] = 2
           return divtime
   # Wait a long while..then normalize
   arr = mandelbrot(4000,4000, maxit=2000)[400:800, 2500:2900]

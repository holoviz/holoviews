"""
Example app demonstrating how to use the HoloViews API to generate
a bokeh app with complex interactivity. Uses a RangeXY stream to allow
interactive exploration of the mandelbrot set.
"""

import numpy as np
import holoviews as hv

from holoviews import Options
from holoviews.plotting.bokeh import BokehRenderer
from holoviews.streams import RangeXY
from numba import jit

@jit
def mandel(x, y, max_iters):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    i = 0
    c = complex(x,y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return 255

@jit
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color

    return image

def get_fractal(x_range, y_range):
    (x0, x1), (y0, y1) = x_range, y_range
    image = np.zeros((600, 600), dtype=np.uint8)
    return hv.Image(create_fractal(x0, x1, -y1, -y0, image, 200),
                    bounds=(x0, y0, x1, y1))

# Define stream linked to axis XY-range
range_stream = RangeXY(x_range=(-1., 1.), y_range=(-1., 1.))

# Create DynamicMap to compute fractal per zoom range and
# adjoin a logarithmic histogram
dmap = hv.DynamicMap(get_fractal, label='Manderbrot Explorer',
                     streams=[range_stream]).hist(log=True)

# Define styling options
options = hv.Store.options('bokeh')
options.Image = {
    'style': Options(cmap='fire'),
    'plot' : Options(logz=True, height=600, width=600,
                     xaxis=None, yaxis=None)
}
options.Histogram = {
    'norm': Options(framewise=True),
    'plot': Options(logy=True, width=200)
}

doc = BokehRenderer.server_doc(dmap)
doc.title = 'Mandelbrot Explorer'

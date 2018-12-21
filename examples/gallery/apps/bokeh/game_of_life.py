import numpy as np
import holoviews as hv

from holoviews import opts
from holoviews.streams import Tap, Counter
from scipy.signal import convolve2d

renderer = hv.renderer('bokeh')

diehard = [[0, 0, 0, 0, 0, 0, 1, 0],
           [1, 1, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 1, 1, 1]]

boat = [[1, 1, 0],
        [1, 0, 1],
        [0, 1, 0]]

r_pentomino = [[0, 1, 1],
               [1, 1, 0],
               [0, 1, 0]]

beacon = [[0, 0, 1, 1],
          [0, 0, 1, 1],
          [1, 1, 0, 0],
          [1, 1, 0, 0]]

acorn = [[0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [1, 1, 0, 0, 1, 1, 1]]

spaceship = [[0, 0, 1, 1, 0],
             [1, 1, 0, 1, 1],
             [1, 1, 1, 1, 0],
             [0, 1, 1, 0, 0]]

block_switch_engine = [[0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 1, 0, 1, 1],
                       [0, 0, 0, 0, 1, 0, 1, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0],
                       [1, 0, 1, 0, 0, 0, 0, 0]]

glider = [[1, 0, 0], [0, 1, 1], [1, 1, 0]]

unbounded = [[1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 1, 1, 0, 1],
            [1, 0, 1, 0, 1]]

shapes = {'Glider': glider, 'Block Switch Engine': block_switch_engine,
          'Spaceship': spaceship, 'Acorn': acorn, 'Beacon': beacon,
          'Diehard': diehard, 'Unbounded': unbounded}

def step(X):
    nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
    return (nbrs_count == 3) | (X & (nbrs_count == 2))

def update(pattern, counter, x, y):
    if x and y:
        pattern = np.array(shapes[pattern])
        r, c = pattern.shape
        y, x = img.sheet2matrixidx(x,y)
        img.data[y:y+r,x:x+c] = pattern[::-1]
    else:
        img.data = step(img.data)
    return hv.Image(img)

title = 'Game of Life - Tap to place pattern, Doubletap to clear'
img_opts = opts.Image(cmap='gray', toolbar=None, height=400, width=800,
                      title=title, xaxis=None, yaxis=None)
img = hv.Image(np.zeros((100, 200), dtype=np.uint8))
counter, tap = Counter(transient=True), Tap(transient=True)
pattern_dim = hv.Dimension('Pattern', values=sorted(shapes.keys()))
dmap = hv.DynamicMap(update, kdims=[pattern_dim], streams=[counter, tap])

doc = renderer.server_doc(dmap.redim.range(z=(0, 1)).opts(img_opts))
dmap.periodic(0.05, None)
doc.title = 'Game of Life'

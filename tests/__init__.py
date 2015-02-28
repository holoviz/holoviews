import os, sys

# Standardize backend due to random inconsistencies
from matplotlib import pyplot
pyplot.switch_backend('agg')

cwd = os.path.abspath(os.path.split(__file__)[0])
sys.path.insert(0, os.path.join(cwd, '..'))

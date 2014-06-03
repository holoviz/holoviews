import os, sys

cwd = os.path.abspath(os.path.split(__file__)[0])
sys.path.insert(0, os.path.join(cwd, '..'))
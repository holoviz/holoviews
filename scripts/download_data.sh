#!/usr/bin/env bash

set -euxo pipefail

bokeh sampledata

python -c "
try:
    import pooch
    import scipy
    import xarray as xr
except ImportError:
    pass
else:
    xr.tutorial.open_dataset('air_temperature')
    xr.tutorial.open_dataset('rasm')
"

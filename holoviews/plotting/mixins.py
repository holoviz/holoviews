from __future__ import absolute_import, division, unicode_literals

import numpy as np

from ..core import util, Dimension


class GeomMixin(object):

    def get_extents(self, element, ranges, range_type='combined'):
        """
        Use first two key dimensions to set names, and all four
        to set the data range.
        """
        kdims = element.kdims
        # loop over start and end points of segments
        # simultaneously in each dimension
        for kdim0, kdim1 in zip([kdims[i].name for i in range(2)],
                                [kdims[i].name for i in range(2,4)]):
            new_range = {}
            for kdim in [kdim0, kdim1]:
                # for good measure, update ranges for both start and end kdim
                for r in ranges[kdim]:
                    # combine (x0, x1) and (y0, y1) in range calculation
                    new_range[r] = util.max_range([ranges[kd][r]
                                                   for kd in [kdim0, kdim1]])
            ranges[kdim0] = new_range
            ranges[kdim1] = new_range
        return super(GeomMixin, self).get_extents(element, ranges, range_type)


class ChordMixin(object):

    def get_extents(self, element, ranges, range_type='combined'):
        """
        A Chord plot is always drawn on a unit circle.
        """
        xdim, ydim = element.nodes.kdims[:2]
        if range_type not in ('combined', 'data', 'extents'):
            return xdim.range[0], ydim.range[0], xdim.range[1], ydim.range[1]
        no_labels = self.labels is None
        rng = 1.1 if no_labels else 1.4
        x0, x1 = util.max_range([xdim.range, (-rng, rng)])
        y0, y1 = util.max_range([ydim.range, (-rng, rng)])
        return (x0, y0, x1, y1)


class SpikesMixin(object):
    
    def get_extents(self, element, ranges, range_type='combined'):
        opts = self.lookup_options(element, 'plot').options
        if len(element.dimensions()) > 1 and 'spike_length' not in opts:
            ydim = element.get_dimension(1)
            s0, s1 = ranges[ydim.name]['soft']
            s0 = min(s0, 0) if util.isfinite(s0) else 0
            s1 = max(s1, 0) if util.isfinite(s1) else 0
            ranges[ydim.name]['soft'] = (s0, s1)
        proxy_dim = None
        if 'spike_length' in opts:
            proxy_dim = Dimension('proxy_dim')
            proxy_range = (self.position, self.position + opts['spike_length'])
            ranges['proxy_dim'] = {'data':    proxy_range,
                                  'hard':     (np.nan, np.nan),
                                  'soft':     (np.nan, np.nan),
                                  'combined': proxy_range}
        l, b, r, t = super(SpikesMixin, self).get_extents(element, ranges, range_type,
                                                          ydim=proxy_dim)
        if len(element.dimensions()) == 1 and range_type != 'hard':
            if self.batched:
                bs, ts = [], []
                # Iterate over current NdOverlay and compute extents
                # from position and length plot options
                frame = self.current_frame or self.hmap.last
                for el in frame.values():
                    opts = self.lookup_options(el, 'plot').options
                    pos = opts.get('position', self.position)
                    length = opts.get('spike_length', self.spike_length)
                    bs.append(pos)
                    ts.append(pos+length)
                b, t = (np.nanmin(bs), np.nanmax(ts))
            else:
                b, t = self.position, self.position+self.spike_length
        return l, b, r, t



class AreaMixin(object):
    
    def get_extents(self, element, ranges, range_type='combined'):
        vdims = element.vdims[:2]
        vdim = vdims[0].name
        if len(vdims) > 1:
            new_range = {}
            for r in ranges[vdim]:
                new_range[r] = util.max_range([ranges[vd.name][r] for vd in vdims])
            ranges[vdim] = new_range
        else:
            s0, s1 = ranges[vdim]['soft']
            s0 = min(s0, 0) if util.isfinite(s0) else 0
            s1 = max(s1, 0) if util.isfinite(s1) else 0
            ranges[vdim]['soft'] = (s0, s1)
        return super(AreaMixin, self).get_extents(element, ranges, range_type)

from collections import defaultdict

from ...core import Store, util
from ...core.options import abbreviated_exception
from .element import ElementPlot
from .chart import line_properties
from .util import rgb2hex


class BatchedElementPlot(ElementPlot):

    _batched = True

    def __init__(self, *args, **kwargs):
        super(BatchedElementPlot, self).__init__(*args, **kwargs)
        self.ordering = util.layer_sort(self.hmap)
        self.style = self.lookup_options(self.hmap.last.last, 'style').max_cycles(len(self.ordering))
        self.offset = 0
        
    def get_style(self, spec):
        if spec not in self.ordering:
            self.ordering = util.layer_sort(self.hmap)
            self.style = self.lookup_options(self.hmap.last.last, 'style').max_cycles(len(self.ordering))
        order = self.ordering.index(spec)
        return self.style[order]


class BatchedCurvePlot(BatchedElementPlot):

    style_opts = ['color'] + line_properties
    _plot_method = 'multi_line'
    _mapping = {p: p for p in ['xs', 'ys', 'color', 'line_alpha']}

    def get_data(self, overlay, ranges=None, empty=False):
        data = defaultdict(list)
        for key, el in overlay.items():
            spec = util.get_overlay_spec(overlay, key, el)
            style = self.get_style(spec)
            for opt in self._mapping:
                if opt in ['xs', 'ys']:
                    index = {'xs': 0, 'ys': 1}[opt]
                    val = el.dimension_values(index)
                else:
                    val = style.get(opt)
                if opt == 'color' and isinstance(val, tuple):
                    val = rgb2hex(val)
                data[opt].append(val)
        data = {opt: vals for opt, vals in data.items()
                if not any(v is None for v in vals)}
        return data, {k: k for k in data}

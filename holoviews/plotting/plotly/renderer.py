import uuid, json
from plotly.offline.offline import utils, get_plotlyjs

import param

from ..renderer import Renderer, MIME_TYPES
from ...core.options import Store
from ...core import HoloMap
from .widgets import PlotlyScrubberWidget, PlotlySelectionWidget


class PlotlyRenderer(Renderer):
    
    backend = 'plotly'

    fig = param.ObjectSelector(default='auto', objects=['html', 'json', 'auto'], doc="""
        Output render format for static figures. If None, no figure
        rendering will occur. """)
    
    mode_formats = {'fig': {'default': ['html', 'json']},
                    'holomap': {'default': ['widgets', 'scrubber', 'auto']}}

    widgets = {'scrubber': PlotlyScrubberWidget,
               'widgets': PlotlySelectionWidget}

    def __call__(self, obj, fmt='html', divuuid=None):
        divuuid = uuid.uuid4() if divuuid is None else divuuid
        plot, fmt =  self._validate(obj, fmt)
        mime_types = {'file-ext':fmt, 'mime_type': MIME_TYPES[fmt]}

        if isinstance(plot, tuple(self.widgets.values())):
            return plot(), mime_types
        elif fmt == 'html':
            return self.figure_data(plot.state, divuuid), mime_types
        elif fmt == 'json':
            return json.dumps({str(divuuid): {'data': plot.state.get('data', []),
                                              'layout': plot.state.get('layout', {})}},
                              cls=utils.PlotlyJSONEncoder), mime_types


    @classmethod
    def figure_data(cls, figure, divuuid=None, width=800, height=600):
        jdata = json.dumps(figure.get('data', []), cls=utils.PlotlyJSONEncoder)
        jlayout = json.dumps(figure.get('layout', {}), cls=utils.PlotlyJSONEncoder)

        config = {}
        config['showLink'] = False
        jconfig = json.dumps(config)
    
        header = ('<script type="text/javascript">'
                  'window.PLOTLYENV=window.PLOTLYENV || {};'
                  '</script>')

        script = '\n'.join([
            'Plotly.plot("{id}", {data}, {layout}, {config}).then(function() {{',
            '    $(".{id}.loading").remove();',
            '}})']).format(id=divuuid,
                           data=jdata,
                           layout=jlayout,
                           config=jconfig)

        content =    ('<div class="{id} loading" style="color: rgb(50,50,50);">'
                      'Drawing...</div>'
                      '<div id="{id}" style="height: {height}; width: {width};" '
                      'class="plotly-graph-div">'
                      '</div>'
                      '<script type="text/javascript">'
                      '{script}'
                      '</script>').format(id=divuuid, script=script,
                                          height=height, width=width)
    
        return '\n'.join([header, content])
    
    
    @classmethod
    def plot_options(cls, obj, percent_size):
        factor = percent_size / 100.0
        obj = obj.last if isinstance(obj, HoloMap) else obj
        plot = Store.registry[cls.backend].get(type(obj), None)
        options = Store.lookup_options(cls.backend, obj, 'plot').options
        width = options.get('width', plot.width) * factor
        height = options.get('height', plot.height) * factor
        return dict(options, **{'width':int(width), 'height': int(height)})


    @classmethod
    def load_nb(cls):
        """
        Loads the plotly notebook resources.
        """
        from IPython.display import display, HTML
        display(HTML(plotly_include()))


def plotly_include():
    return """
            <script type="text/javascript">
            require_=require;requirejs_=requirejs; define_=define;
            require=requirejs=define=undefined;
            </script>
            <script type="text/javascript">
            {include}
            </script>
            <script type="text/javascript">
            require=require_;requirejs=requirejs_; define=define_;
            </script>""".format(include=get_plotlyjs())


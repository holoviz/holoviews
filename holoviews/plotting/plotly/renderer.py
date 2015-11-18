import uuid, json
from plotly.offline.offline import utils

from holoviews.plotting.renderer import Renderer, MIME_TYPES
from holoviews.core.options import Store
from holoviews.core import HoloMap

def render_plotly(figure, width=800, height=600):
    plotdivid = uuid.uuid4()
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
        '}})'
    ]).format(id=plotdivid,
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
                 '</script>').format(id=plotdivid, script=script,
                                   height=height, width=width)
    
    return '\n'.join([header, content])

class PlotlyRenderer(Renderer):
    
    backend = 'plotly'
    
    mode_formats = {'fig': {'default': ['html']},
                    'holomap': {'default': [None]}}
    
    
    def __call__(self, obj, fmt='html'):
        return render_plotly(obj.state), {'file-ext':fmt, 'mime_type': MIME_TYPES[fmt]}
    
    
    @classmethod
    def plot_options(cls, obj, percent_size):
        factor = percent_size / 100.0
        obj = obj.last if isinstance(obj, HoloMap) else obj
        plot = Store.registry[cls.backend].get(type(obj), None)
        options = Store.lookup_options(cls.backend, obj, 'plot').options
        width = options.get('width', plot.width) * factor
        height = options.get('height', plot.height) * factor
        return dict(options, **{'width':int(width), 'height': int(height)})


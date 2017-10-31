import datetime as dt

import psutil
import pandas as pd
import holoviews as hv

hv.extension('bokeh')

# Define functions to get memory and CPU usage
def get_mem_data():
    vmem = psutil.virtual_memory()
    df = pd.DataFrame(dict(free=vmem.free/vmem.total,
                           used=vmem.used/vmem.total),
                      index=[pd.Timestamp.now()])
    return df*100

def get_cpu_data():
    cpu_percent = psutil.cpu_percent(percpu=True)
    df = pd.DataFrame(list(enumerate(cpu_percent)),
                      columns=['CPU', 'Utilization'])
    df['time'] = pd.Timestamp.now()
    return df


# Define DynamicMap callbacks returning Elements

def mem_stack(data):
    data = pd.melt(data, 'index', var_name='Type', value_name='Usage')
    areas = hv.Dataset(data).to(hv.Area, 'index', 'Usage')
    return hv.Area.stack(areas.overlay()).relabel('Memory')

def cpu_box(data):
    return hv.BoxWhisker(data, 'CPU', 'Utilization').relabel('CPU Usage')


# Set up StreamingDataFrame and add async callback

cpu_stream = hv.streams.Buffer(get_cpu_data(), 800, index=False)
mem_stream = hv.streams.Buffer(get_mem_data())

def cb():
    cpu_stream.send(get_cpu_data())
    mem_stream.send(get_mem_data())


# Define DynamicMaps and display plot

cpu_dmap = hv.DynamicMap(cpu_box, streams=[cpu_stream])
mem_dmap = hv.DynamicMap(mem_stack, streams=[mem_stream])

cpu_opts = {'plot': dict(width=500, height=400, color_index='CPU'),
        'style': dict(box_fill_color=hv.Cycle('Category20'))}
mem_opts = dict(height=400, width=400)

plot = (cpu_dmap.redim.range(Utilization=(0, 100)).opts(**cpu_opts) +
        mem_dmap.redim.range(Usage=(0, 100)).opts(plot=mem_opts))


# Render plot and attach periodic callback

doc = hv.renderer('bokeh').server_doc(plot)
doc.add_periodic_callback(cb, 0.05)

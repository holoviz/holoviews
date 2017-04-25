import plotly.graph_objs as go


def add_figure(fig, subfig, r, c, idx):
    """
    Combines a figure with an existing figure created with
    plotly.tools.make_subplots, by adding the data and merging
    axis layout options.
    """
    ref = fig._grid_ref[r][c][0][1:]
    layout = replace_refs(subfig['layout'], ref)

    fig['layout']['xaxis%s'%ref].update(layout.get('xaxis', {}))
    fig['layout']['yaxis%s'%ref].update(layout.get('yaxis', {}))
    fig['layout']['annotations'].extend(layout.get('annotations', []))
    for d in subfig['data']:
        fig.append_trace(d, r+1, c+1)


def replace_refs(obj, ind):
    """
    Replaces xref and yref to allow combining multiple plots
    """
    if isinstance(obj, go.graph_objs.PlotlyList):
        return [replace_refs(o, ind) for o in obj]
    elif isinstance(obj, go.graph_objs.PlotlyDict):
        new_obj = {}
        for k, v in obj.items():
            if k in ['xref', 'yref']:
                v = '{ax}{ind}'.format(ax=k[0], ind=ind)
            new_obj[k] = replace_refs(v, ind)
        return new_obj
    else:
        return obj

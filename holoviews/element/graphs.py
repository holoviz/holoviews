from types import FunctionType

import param
import numpy as np

from ..core import Dimension, Dataset, Element2D
from ..core.dimension import redim
from ..core.util import max_range
from ..core.operation import Operation
from .chart import Points
from .path import Path
from .util import split_path, pd

try:
    from datashader.layout import LayoutAlgorithm as ds_layout
except:
    ds_layout = None


class redim_graph(redim):
    """
    Extension for the redim utility that allows re-dimensioning
    Graph objects including their nodes and edgepaths.
    """

    def __call__(self, specs=None, **dimensions):
        redimmed = super(redim_graph, self).__call__(specs, **dimensions)
        new_data = (redimmed.data,)
        if self.parent.nodes:
            new_data = new_data + (self.parent.nodes.redim(specs, **dimensions),)
        if self.parent._edgepaths:
            new_data = new_data + (self.parent.edgepaths.redim(specs, **dimensions),)
        return redimmed.clone(new_data)


def circular_layout(nodes):
    N = len(nodes)
    circ = np.pi/N*np.arange(N)*2
    x = np.cos(circ)
    y = np.sin(circ)
    return (x, y, nodes)


class layout_nodes(Operation):
    """
    Accepts a Graph and lays out the corresponding nodes with the
    supplied networkx layout function. If no layout function is
    supplied uses a simple circular_layout function. Also supports
    LayoutAlgorithm function provided in datashader layouts.
    """

    only_nodes = param.Boolean(default=False, doc="""
        Whether to return Nodes or Graph.""")

    layout = param.Callable(default=None, doc="""
        A NetworkX layout function""")

    kwargs = param.Dict(default={}, doc="""
        Keyword arguments passed to the layout function.""")

    def _process(self, element, key=None):
        if self.p.layout and isinstance(self.p.layout, FunctionType):
            import networkx as nx
            graph = nx.from_edgelist(element.array([0, 1]))
            positions = self.p.layout(graph, **self.p.kwargs)
            nodes = [tuple(pos)+(idx,) for idx, pos in sorted(positions.items())]
        else:
            source = element.dimension_values(0, expanded=False)
            target = element.dimension_values(1, expanded=False)
            nodes = np.unique(np.concatenate([source, target]))
            if self.p.layout:
                import pandas as pd
                df = pd.DataFrame({'index': nodes})
                nodes = self.p.layout(df, element.dframe(), **self.p.kwargs)
                nodes = nodes[['x', 'y', 'index']]
            else:
                nodes = circular_layout(nodes)
        if self.p.only_nodes:
            return Nodes(nodes)
        return element.clone((element.data, nodes))
        


class Graph(Dataset, Element2D):
    """
    Graph is high-level Element representing both nodes and edges.
    A Graph may be defined in an abstract form representing just
    the abstract edges between nodes and optionally may be made
    concrete by supplying a Nodes Element defining the concrete
    positions of each node. If the node positions are supplied
    the EdgePaths (defining the concrete edges) can be inferred
    automatically or supplied explicitly.

    The constructor accepts regular columnar data defining the edges
    or a tuple of the abstract edges and nodes, or a tuple of the
    abstract edges, nodes, and edgepaths.
    """

    group = param.String(default='Graph', constant=True)

    kdims = param.List(default=[Dimension('start'), Dimension('end')],
                       bounds=(2, 2))

    def __init__(self, data, kdims=None, vdims=None, **params):
        if isinstance(data, tuple):
            data = data + (None,)* (3-len(data))
            edges, nodes, edgepaths = data
        else:
            edges, nodes, edgepaths = data, None, None
        if nodes is not None:
            node_info = None
            if isinstance(nodes, Nodes):
                pass
            elif not isinstance(nodes, Dataset) or nodes.ndims == 3:
                nodes = Nodes(nodes)
            else:
                node_info = nodes
                nodes = None
        else:
            node_info = None
        if edgepaths is not None and not isinstance(edgepaths, EdgePaths):
            edgepaths = EdgePaths(edgepaths)
        self._nodes = nodes
        self._edgepaths = edgepaths
        super(Graph, self).__init__(edges, kdims=kdims, vdims=vdims, **params)
        if self._nodes is None and node_info:
            nodes = self.nodes.clone(datatype=['pandas', 'dictionary'])
            for d in node_info.dimensions():
                nodes = nodes.add_dimension(d, len(nodes.vdims),
                                            node_info.dimension_values(d),
                                            vdim=True)
            self._nodes = nodes
        self._validate()
        self.redim = redim_graph(self, mode='dataset')


    def _validate(self):
        if self._edgepaths is None:
            return
        mismatch = []
        for kd1, kd2 in zip(self.nodes.kdims, self.edgepaths.kdims):
            if kd1 != kd2:
                mismatch.append('%s != %s' % (kd1, kd2))
        if mismatch:
            raise ValueError('Ensure that the first two key dimensions on '
                             'Nodes and EdgePaths match: %s' % ', '.join(mismatch))
        npaths = len(self._edgepaths.data)
        nedges = len(self)
        if nedges != npaths:
            mismatch = True
            if npaths == 1:
                edges = self.edgepaths.split()[0]
                vals = edges.dimension_values(0)
                npaths = len(np.where(np.isnan(vals))[0])
                if not np.isnan(vals[-1]):
                    npaths += 1
                mismatch = npaths != nedges
            if mismatch:
                raise ValueError('Ensure that the number of edges supplied '
                                 'to the Graph (%d) matches the number of '
                                 'edgepaths (%d)' % (nedges, npaths))


    def clone(self, data=None, shared_data=True, new_type=None, *args, **overrides):
        if data is None:
            data = (self.data, self.nodes)
            if self._edgepaths:
                data = data + (self.edgepaths,)
            overrides['plot_id'] = self._plot_id
        elif not isinstance(data, tuple):
            data = (data, self.nodes)
            if self._edgepaths:
                data = data + (self.edgepaths,)
        return super(Graph, self).clone(data, shared_data, new_type, *args, **overrides)


    def select(self, selection_specs=None, selection_mode='edges', **selection):
        """
        Allows selecting data by the slices, sets and scalar values
        along a particular dimension. The indices should be supplied as
        keywords mapping between the selected dimension and
        value. Additionally selection_specs (taking the form of a list
        of type.group.label strings, types or functions) may be
        supplied, which will ensure the selection is only applied if the
        specs match the selected object.

        Selecting by a node dimensions selects all edges and nodes that are
        connected to the selected nodes. To select only edges between the
        selected nodes set the selection_mode to 'nodes'.
        """
        selection = {dim: sel for dim, sel in selection.items()
                     if dim in self.dimensions('ranges')+['selection_mask']}
        if (selection_specs and not any(self.matches(sp) for sp in selection_specs)
            or not selection):
            return self

        index_dim = self.nodes.kdims[2].name
        dimensions = self.kdims+self.vdims
        node_selection = {index_dim: v for k, v in selection.items()
                          if k in self.kdims}
        nodes = self.nodes.select(**dict(selection, **node_selection))
        selection = {k: v for k, v in selection.items() if k in dimensions}

        # Compute mask for edges if nodes were selected on
        nodemask = None
        if len(nodes) != len(self.nodes):
            xdim, ydim = dimensions[:2]
            indices = list(nodes.dimension_values(2, False))
            if selection_mode == 'edges':
                mask1 = self.interface.select_mask(self, {xdim.name: indices})
                mask2 = self.interface.select_mask(self, {ydim.name: indices})
                nodemask = (mask1 | mask2)
                nodes = self.nodes
            else:
                nodemask = self.interface.select_mask(self, {xdim.name: indices,
                                                             ydim.name: indices})

        # Compute mask for edge selection
        mask = None
        if selection:
            mask = self.interface.select_mask(self, selection)

        # Combine masks
        if nodemask is not None:
            if mask is not None:
                mask &= nodemask
            else:
                mask = nodemask

        # Apply edge mask
        if mask is not None:
            data = self.interface.select(self, mask)
            if not np.all(mask):
                new_graph = self.clone((data, nodes))
                source = new_graph.dimension_values(0, expanded=False)
                target = new_graph.dimension_values(1, expanded=False)
                unique_nodes = np.unique(np.concatenate([source, target]))
                nodes = new_graph.nodes[:, :, list(unique_nodes)]
            paths = None
            if self._edgepaths:
                edgepaths = self._split_edgepaths
                paths = edgepaths.clone(edgepaths.interface.select_paths(edgepaths, mask))
                if len(self._edgepaths.data) == 1:
                    paths = paths.clone([paths.dframe() if pd else paths.array()])
        else:
            data = self.data
            paths = self._edgepaths
        return self.clone((data, nodes, paths))


    @property
    def _split_edgepaths(self):
        if len(self) == len(self._edgepaths.data):
            return self._edgepaths
        else:
            return self._edgepaths.clone(split_path(self._edgepaths))


    def range(self, dimension, data_range=True):
        if self.nodes and dimension in self.nodes.dimensions():
            node_range = self.nodes.range(dimension, data_range)
            if self._edgepaths:
                path_range = self._edgepaths.range(dimension, data_range)
                return max_range([node_range, path_range])
            return node_range
        return super(Graph, self).range(dimension, data_range)


    def dimensions(self, selection='all', label=False):
        dimensions = super(Graph, self).dimensions(selection, label)
        if selection == 'ranges':
            if self._nodes:
                node_dims = self.nodes.dimensions(selection, label)
            else:
                node_dims = Nodes.kdims+Nodes.vdims
                if label in ['name', True, 'short']:
                    node_dims = [d.name for d in node_dims]
                elif label in ['long', 'label']:
                    node_dims = [d.label for d in node_dims]
            return dimensions+node_dims
        return dimensions


    @property
    def nodes(self):
        """
        Computes the node positions the first time they are requested
        if no explicit node information was supplied.
        """
        if self._nodes is None:
            self._nodes = layout_nodes(self, only_nodes=True)
        return self._nodes


    @property
    def edgepaths(self):
        """
        Returns the fixed EdgePaths or computes direct connections
        between supplied nodes.
        """
        if self._edgepaths:
            return self._edgepaths
        paths = []
        for start, end in self.array(self.kdims):
            start_ds = self.nodes[:, :, start]
            end_ds = self.nodes[:, :, end]
            if not len(start_ds) or not len(end_ds):
                raise ValueError('Could not find node positions for all edges')
            sx, sy = start_ds.array(start_ds.kdims[:2]).T
            ex, ey = end_ds.array(end_ds.kdims[:2]).T
            paths.append([(sx[0], sy[0]), (ex[0], ey[0])])
        return EdgePaths(paths, kdims=self.nodes.kdims[:2])


    @classmethod
    def from_networkx(cls, G, layout_function, nodes=None, **kwargs):
        """
        Generate a HoloViews Graph from a networkx.Graph object and
        networkx layout function. Any keyword arguments will be passed
        to the layout function.
        """
        positions = layout_function(G, **kwargs)
        edges = G.edges()
        if nodes:
            idx_dim = nodes.kdims[-1].name
            xs, ys = zip(*[v for k, v in sorted(positions.items())])
            indices = list(nodes.dimension_values(idx_dim))
            edges = [(src, tgt) for (src, tgt) in edges if src in indices and tgt in indices]
            nodes = nodes.select(**{idx_dim: [eid for e in edges for eid in e]}).sort()
            nodes = nodes.add_dimension('x', 0, xs)
            nodes = nodes.add_dimension('y', 1, ys).clone(new_type=Nodes)
        else:
            nodes = Nodes([tuple(pos)+(idx,) for idx, pos in sorted(positions.items())])
        return cls((edges, nodes))


class Nodes(Points):
    """
    Nodes is a simple Element representing Graph nodes as a set of
    Points.  Unlike regular Points, Nodes must define a third key
    dimension corresponding to the node index.
    """

    kdims = param.List(default=[Dimension('x'), Dimension('y'),
                                Dimension('index')], bounds=(3, 3))

    group = param.String(default='Nodes', constant=True)


class EdgePaths(Path):
    """
    EdgePaths is a simple Element representing the paths of edges
    connecting nodes in a graph.
    """

    group = param.String(default='EdgePaths', constant=True)


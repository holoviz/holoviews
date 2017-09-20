import param
import numpy as np

from ..core import Dimension, Dataset, Element2D
from ..core.dimension import redim
from ..core.util import max_range
from ..core.operation import Operation
from .chart import Points
from .path import Path

class graph_redim(redim):
    """
    Extension for the redim utility that allows re-dimensioning
    Graph objects including their nodes and edgepaths.
    """

    def __call__(self, specs=None, **dimensions):
        redimmed = super(graph_redim, self).__call__(specs, **dimensions)
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
    supplied uses a simple circular_layout function.
    """

    layout = param.Callable(default=None, doc="""
        A NetworkX layout function""")

    def _process(self, element, key=None):
        if self.p.layout:
            graph = nx.from_edgelist(element.array([0, 1]))
            positions = self.p.layout(graph)
            return Nodes([tuple(pos)+(idx,) for idx, pos in sorted(positions.items())])
        else:
            source = element.dimension_values(0, expanded=False)
            target = element.dimension_values(1, expanded=False)
            nodes = np.unique(np.concatenate([source, target]))
            return Nodes(circular_layout(nodes))


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

    def __init__(self, data, **params):
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
        super(Graph, self).__init__(edges, **params)
        if self._nodes is None and node_info:
            nodes = self.nodes.clone(datatype=['pandas', 'dictionary'])
            for d in node_info.dimensions():
                nodes = nodes.add_dimension(d, len(nodes.vdims),
                                            node_info.dimension_values(d),
                                            vdim=True)
            self._nodes = nodes
        if self._edgepaths:
            mismatch = []
            for kd1, kd2 in zip(self.nodes.kdims, self.edgepaths.kdims):
                if kd1 != kd2:
                    mismatch.append('%s != %s' % (kd1, kd2))
            if mismatch:
                raise ValueError('Ensure that the first two key dimensions on '
                                 'Nodes and EdgePaths match: %s' % ', '.join(mismatch))
        self.redim = graph_redim(self, mode='dataset')


    def clone(self, data=None, shared_data=True, new_type=None, *args, **overrides):
        if data is None:
            data = (self.data, self.nodes)
            if self._edgepaths:
                data = data + (self.edgepaths,)
        elif not isinstance(data, tuple):
            data = (data, self.nodes)
            if self._edgepaths:
                data = data + (self.edgepaths,)
        return super(Graph, self).clone(data, shared_data, new_type, *args, **overrides)


    def select(self, selection_specs=None, **selection):
        """
        Allows selecting data by the slices, sets and scalar values
        along a particular dimension. The indices should be supplied as
        keywords mapping between the selected dimension and
        value. Additionally selection_specs (taking the form of a list
        of type.group.label strings, types or functions) may be
        supplied, which will ensure the selection is only applied if the
        specs match the selected object.
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
        if len(nodes) != len(self):
            xdim, ydim = dimensions[:2]
            indices = list(nodes.dimension_values(2))
            selection[xdim.name] = indices
            selection[ydim.name] = indices
        if selection:
            mask = self.interface.select_mask(self, selection)
            data = self.interface.select(self, mask)
            if self._edgepaths:
                paths = self.edgepaths.interface.select_paths(self.edgepaths, mask)
                return self.clone((data, nodes, paths))
        else:
            data = self.data
            if self._edgepaths:
                return self.clone((data, nodes, self._edgepaths))
        return self.clone((data, nodes))


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
            self._nodes = layout_nodes(self)
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
        if nodes:
            xs, ys = zip(*[v for k, v in sorted(positions.items())])
            nodes = nodes.add_dimension('x', 0, xs)
            nodes = nodes.add_dimension('y', 1, ys).clone(new_type=Nodes)
        else:
            nodes = Nodes([tuple(pos)+(idx,) for idx, pos in sorted(positions.items())])
        return cls((G.edges(), nodes))


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

import param

from ..core import Dimension, Dataset, Element2D
from ..core.dimension import redim
from ..core.util import max_range
from .chart import Points
from .path import Path

class graph_redim(redim):
    """
    Extension for the redim utility that allows re-dimensioning
    Graph objects including their nodes and nodepaths.
    """

    def __call__(self, specs=None, **dimensions):
        redimmed = super(graph_redim, self).__call__(specs, **dimensions)
        new_data = (redimmed.data,)
        if self.parent.nodes:
            new_data = new_data + (self.parent.nodes.redim(specs, **dimensions),)
        if self.parent._nodepaths:
            new_data = new_data + (self.parent.nodepaths.redim(specs, **dimensions),)
        return redimmed.clone(new_data)


class Graph(Dataset, Element2D):
    """
    Graph is high-level Element representing both nodes and edges.
    A Graph may be defined in an abstract form representing just
    the abstract edges between nodes and optionally may be made
    concrete by supplying a Nodes Element defining the concrete
    positions of each node. If the node positions are supplied
    the NodePaths (defining the concrete edges) can be inferred
    automatically or supplied explicitly.

    The constructor accepts regular columnar data defining the edges
    or a tuple of the abstract edges and nodes, or a tuple of the
    abstract edges, nodes, and nodepaths.
    """

    group = param.String(default='Graph')

    kdims = param.List(default=[Dimension('start'), Dimension('end')],
                       bounds=(2, 2))

    def __init__(self, data, **params):
        if isinstance(data, tuple):
            data = data + (None,)* (3-len(data))
            edges, nodes, nodepaths = data
        else:
            edges, nodes, nodepaths = data, None, None
        if nodes is not None and not isinstance(nodes, Nodes):
            nodes = Nodes(nodes)
        if nodepaths is not None and not isinstance(nodepaths, NodePaths):
            nodepaths = NodePaths(nodepaths)
        self.nodes = nodes
        self._nodepaths = nodepaths
        super(Graph, self).__init__(edges, **params)
        self.redim = graph_redim(self, mode='dataset')

    def clone(self, data=None, shared_data=True, new_type=None, *args, **overrides):
        if data is None:
            data = (self.data, self.nodes)
            if self._nodepaths:
                data = data + (self.nodepaths,)
        elif not isinstance(data, tuple):
            data = (data, self.nodes)
            if self._nodepaths:
                data = data + (self.nodepaths,)
        return super(Graph, self).clone(data, shared_data, new_type, *args, **overrides)


    def range(self, dimension, data_range=True):
        if self.nodes and dimension in self.nodes.dimensions():
            node_range = self.nodes.range(dimension, data_range)
            if self._nodepaths:
                path_range = self._nodepaths.range(dimension, data_range)
                return max_range([node_range, path_range])
            return node_range
        return super(Graph, self).range(dimension, data_range)


    def dimensions(self, selection='all', label=False):
        dimensions = super(Graph, self).dimensions(selection, label)
        if self.nodes and selection == 'all':
            return dimensions+self.nodes.dimensions(selection, label)
        return dimensions


    @property
    def nodepaths(self):
        """
        Returns the fixed NodePaths or computes direct connections
        between supplied nodes.
        """
        if self.nodes is None:
            raise ValueError('Cannot return NodePaths without node positions')
        elif self._nodepaths:
            return self._nodepaths
        paths = []
        for start, end in self.array(self.kdims):
            start_ds = self.nodes[:, :, start]
            end_ds = self.nodes[:, :, end]
            sx, sy = start_ds.array(start_ds.kdims[:2]).T
            ex, ey = end_ds.array(end_ds.kdims[:2]).T
            paths.append([(sx[0], sy[0]), (ex[0], ey[0])])
        return NodePaths(paths)

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

    group = param.String(default='Nodes')


class NodePaths(Path):
    """
    NodePaths is a simple Element representing the paths of edges
    connecting nodes in a graph.
    """

    group = param.String(default='NodePaths')

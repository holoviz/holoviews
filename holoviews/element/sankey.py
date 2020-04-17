from __future__ import division

from collections import Counter
from functools import cmp_to_key
from itertools import cycle

import param
import numpy as np

from ..core.dimension import Dimension
from ..core.data import Dataset
from ..core.operation import Operation
from ..core.util import OrderedDict, unique_array, RecursionError, get_param_values
from .graphs import Graph, Nodes, EdgePaths
from .util import quadratic_bezier


class _layout_sankey(Operation):
    """
    Computes a Sankey diagram from a Graph element for internal use in
    the Sankey element constructor.

    Adapted from d3-sankey under BSD-3 license.
    """

    bounds = param.NumericTuple(default=(0, 0, 1000, 500))

    node_width = param.Number(default=15, doc="""
        Width of the nodes.""")

    node_padding = param.Integer(default=None, allow_None=True, doc="""
        Number of pixels of padding relative to the bounds.""")

    iterations = param.Integer(default=32, doc="""
        Number of iterations to run the layout algorithm.""")

    node_sort = param.Boolean(default=True, doc="""
        Sort nodes in ascending breadth.""")

    def _process(self, element, key=None):
        nodes, edges, graph = self.layout(element, **self.p)
        params = get_param_values(element)
        return Sankey((element.data, nodes, edges), sankey=graph, **params)

    def layout(self, element, **params):
        self.p = param.ParamOverrides(self, params)
        graph = {'nodes': [], 'links': []}
        self.computeNodeLinks(element, graph)
        self.computeNodeValues(graph)
        self.computeNodeDepths(graph)
        self.computeNodeBreadths(graph)
        self.computeLinkBreadths(graph)
        paths = self.computePaths(graph)

        node_data = []
        for node in graph['nodes']:
            node_data.append((np.mean([node['x0'], node['x1']]),
                              np.mean([node['y0'], node['y1']]),
                              node['index'])+tuple(node['values']))
        if element.nodes.ndims == 3:
            kdims = element.nodes.kdims
        elif element.nodes.ndims:
            kdims = element.node_type.kdims[:2] + element.nodes.kdims[-1:]
        else:
            kdims = element.node_type.kdims
        nodes = element.node_type(node_data, kdims=kdims, vdims=element.nodes.vdims)
        edges = element.edge_type(paths)
        return nodes, edges, graph


    def computePaths(self, graph):
        paths = []
        for link in graph['links']:
            source, target = link['source'], link['target']
            x0, y0 = source['x1'], link['y0']
            x1, y1 = target['x0'], link['y1']
            start = np.array([(x0, link['width']+y0),
                              (x0, y0)])
            src = (x0, y0)
            ctr1 = ((x0+x1)/2., y0)
            ctr2 = ((x0+x1)/2., y1)
            tgt = (x1, y1)
            bottom = quadratic_bezier(src, tgt, ctr1, ctr2)
            mid = np.array([(x1, y1),
                            (x1, y1+link['width'])])

            xmid = (x0+x1)/2.
            y0 = y0+link['width']
            y1 = y1+link['width']
            src = (x1, y1)
            ctr1 = (xmid, y1)
            ctr2 = (xmid, y0)
            tgt = (x0, y0)
            top = quadratic_bezier(src, tgt, ctr1, ctr2)
            spline = np.concatenate([start, bottom, mid, top])
            paths.append(spline)
        return paths


    @classmethod
    def weightedSource(cls, link):
        return cls.nodeCenter(link['source']) * link['value']

    @classmethod
    def weightedTarget(cls, link):
        return cls.nodeCenter(link['target']) * link['value']

    @classmethod
    def nodeCenter(cls, node):
        return (node['y0'] + node['y1']) / 2

    @classmethod
    def ascendingBreadth(cls, a, b):
        return int(a['y0'] - b['y0'])

    @classmethod
    def ascendingSourceBreadth(cls, a, b):
        return cls.ascendingBreadth(a['source'], b['source']) | a['index'] - b['index']

    @classmethod
    def ascendingTargetBreadth(cls, a, b):
        return cls.ascendingBreadth(a['target'], b['target']) | a['index'] - b['index']

    @classmethod
    def computeNodeLinks(cls, element, graph):
        """
        Populate the sourceLinks and targetLinks for each node.
        Also, if the source and target are not objects, assume they are indices.
        """
        index = element.nodes.kdims[-1]
        node_map = {}
        if element.nodes.vdims:
            values = zip(*(element.nodes.dimension_values(d)
                           for d in element.nodes.vdims))
        else:
            values = cycle([tuple()])
        for index, vals in zip(element.nodes.dimension_values(index), values):
            node = {'index': index, 'sourceLinks': [], 'targetLinks': [], 'values': vals}
            graph['nodes'].append(node)
            node_map[index] = node

        links = [element.dimension_values(d) for d in element.dimensions()[:3]]
        for i, (src, tgt, value) in enumerate(zip(*links)):
            source, target = node_map[src], node_map[tgt]
            link = dict(index=i, source=source, target=target, value=value)
            graph['links'].append(link)
            source['sourceLinks'].append(link)
            target['targetLinks'].append(link)

    @classmethod
    def computeNodeValues(cls, graph):
        """
        Compute the value (size) of each node by summing the associated links.
        """
        for node in graph['nodes']:
            source_val = np.sum([l['value'] for l in node['sourceLinks']])
            target_val = np.sum([l['value'] for l in node['targetLinks']])
            node['value'] = max([source_val, target_val])

    def computeNodeDepths(self, graph):
        """
        Iteratively assign the depth (x-position) for each node.
        Nodes are assigned the maximum depth of incoming neighbors plus one;
        nodes with no incoming links are assigned depth zero, while
        nodes with no outgoing links are assigned the maximum depth.
        """
        nodes = graph['nodes']
        depth = 0
        while nodes:
            next_nodes = []
            for node in nodes:
                node['depth'] = depth
                for link in node['sourceLinks']:
                    if not any(link['target'] is node for node in next_nodes):
                        next_nodes.append(link['target'])
            nodes = next_nodes
            depth += 1
            if depth > 10000:
                raise RecursionError('Sankey diagrams only support acyclic graphs.')

        nodes = graph['nodes']
        depth = 0
        while nodes:
            next_nodes = []
            for node in nodes:
                node['height'] = depth
                for link in node['targetLinks']:
                    if not any(link['source'] is node for node in next_nodes):
                        next_nodes.append(link['source'])
            nodes = next_nodes
            depth += 1
            if depth > 10000:
                raise RecursionError('Sankey diagrams only support acyclic graphs.')

        x0, _, x1, _ = self.p.bounds
        dx = self.p.node_width
        kx = (x1 - x0 - dx) / (depth - 1)
        for node in graph['nodes']:
            d  = node['depth'] if node['sourceLinks'] else depth - 1
            node['x0'] = x0 + max([0, min([depth-1, np.floor(d)]) * kx])
            node['x1'] = node['x0'] + dx

    def computeNodeBreadths(self, graph):
        node_map = OrderedDict()
        depths = Counter()
        for n in graph['nodes']:
            if n['x0'] not in node_map:
                node_map[n['x0']] = []
            node_map[n['x0']].append(n)
            depths[n['depth']] += 1

        _, y0, _, y1 = self.p.bounds
        py = self.p.node_padding
        if py is None:
            max_depth = max(depths.values()) - 1 if depths else 1
            height = self.p.bounds[3] - self.p.bounds[1]
            py = min((height * 0.1) / max_depth, 20) if max_depth else 20

        def initializeNodeBreadth():
            kys = []
            for nodes in node_map.values():
                nsum = np.sum([node['value'] for node in nodes])
                ky = (y1 - y0 - (len(nodes)-1) * py) / nsum
                kys.append(ky)
            ky = np.min(kys) if len(kys) else np.nan

            for nodes in node_map.values():
                for i, node in enumerate(nodes):
                    node['y0'] = i
                    node['y1'] = i + node['value'] * ky

            for link in graph['links']:
                link['width'] = link['value'] * ky

        def relaxLeftToRight(alpha):
            for nodes in node_map.values():
                for node in nodes:
                    if not node['targetLinks']:
                        continue
                    weighted = sum([self.weightedSource(l) for l in node['targetLinks']])
                    tsum = sum([l['value'] for l in node['targetLinks']])
                    center = self.nodeCenter(node)
                    dy = (weighted/tsum - center)*alpha
                    node['y0'] += dy
                    node['y1'] += dy


        def relaxRightToLeft(alpha):
            for nodes in list(node_map.values())[::-1]:
                for node in nodes:
                    if not node['sourceLinks']:
                        continue
                    weighted = sum([self.weightedTarget(l) for l in node['sourceLinks']])
                    tsum = sum([l['value'] for l in node['sourceLinks']])
                    center = self.nodeCenter(node)
                    dy = (weighted/tsum - center)*alpha
                    node['y0'] += dy
                    node['y1'] += dy


        def resolveCollisions():
            for nodes in node_map.values():
                y = y0
                if self.p.node_sort:
                    nodes.sort(key=cmp_to_key(self.ascendingBreadth))
                for node in nodes:
                    dy = y-node['y0']
                    if dy > 0:
                        node['y0'] += dy
                        node['y1'] += dy
                    y = node['y1'] + py

                dy = y-py-y1
                if dy > 0:
                    node['y0'] -= dy
                    node['y1'] -= dy
                    y = node['y0']
                    for node in nodes[:-1][::-1]:
                        dy = node['y1'] + py - y;
                        if dy>0:
                            node['y0'] -= dy
                            node['y1'] -= dy
                        y = node['y0']

        initializeNodeBreadth()
        resolveCollisions()
        alpha = 1
        for _ in range(self.p.iterations):
            alpha = alpha*0.99
            relaxRightToLeft(alpha)
            resolveCollisions()
            relaxLeftToRight(alpha)
            resolveCollisions()

    @classmethod
    def computeLinkBreadths(cls, graph):
        for node in graph['nodes']:
            node['sourceLinks'].sort(key=cmp_to_key(cls.ascendingTargetBreadth))
            node['targetLinks'].sort(key=cmp_to_key(cls.ascendingSourceBreadth))

        for node in graph['nodes']:
            y0 = y1 = node['y0']
            for link in node['sourceLinks']:
                link['y0'] = y0
                y0 += link['width']
            for link in node['targetLinks']:
                link['y1'] = y1
                y1 += link['width']



class Sankey(Graph):
    """
    Sankey is an acyclic, directed Graph type that represents the flow
    of some quantity between its nodes.
    """

    group = param.String(default='Sankey', constant=True)

    vdims = param.List(default=[Dimension('Value')])

    def __init__(self, data, kdims=None, vdims=None, **params):
        if data is None:
            data = []
        if isinstance(data, tuple):
            data = data + (None,)*(3-len(data))
            edges, nodes, edgepaths = data
        else:
            edges, nodes, edgepaths = data, None, None
        sankey_graph = params.pop('sankey', None)
        compute = not (sankey_graph and isinstance(nodes, Nodes) and isinstance(edgepaths, EdgePaths))
        super(Graph, self).__init__(edges, kdims=kdims, vdims=vdims, **params)
        if compute:
            if nodes is None:
                src = self.dimension_values(0, expanded=False)
                tgt = self.dimension_values(1, expanded=False)
                values = unique_array(np.concatenate([src, tgt]))
                nodes = Dataset(values, 'index')
            elif not isinstance(nodes, Dataset):
                try:
                    nodes = Dataset(nodes)
                except:
                    nodes = Dataset(nodes, 'index')
            if not nodes.kdims:
                raise ValueError('Could not determine index in supplied node data. '
                                 'Ensure data has at least one key dimension, '
                                 'which matches the node ids on the edges.')
            self._nodes = nodes
            nodes, edgepaths, graph = _layout_sankey.instance().layout(self)
            self._nodes = nodes
            self._edgepaths = edgepaths
            self._sankey = graph
        else:
            if not isinstance(nodes, self.node_type):
                raise TypeError("Expected Nodes object in data, found %s."
                                % type(nodes))
            self._nodes = nodes
            if not isinstance(edgepaths, self.edge_type):
                raise TypeError("Expected EdgePaths object in data, found %s."
                                % type(edgepaths))
            self._edgepaths = edgepaths
            self._sankey = sankey_graph
        self._validate()

    def clone(self, data=None, shared_data=True, new_type=None, link=True,
              *args, **overrides):
        if data is None:
            overrides['sankey'] = self._sankey
        return super(Sankey, self).clone(data, shared_data, new_type, link,
                                         *args, **overrides)

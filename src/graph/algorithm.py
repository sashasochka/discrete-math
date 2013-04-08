"""
Different graph algorithm implementations
Works only with zero-based graphs
"""
import sys
import heapq
from graph.weighted import Graph as WeightedGraph
from graph.weighted.directed import Graph as WeightedDirectedGraph


undefined_node = -1


class PathSearchNode:
    """
    Represents a node for distance searching
    """

    def __init__(self, distance: int, child: int, parent: int):
        """
        Initialize public fields from arguments
        """
        self.distance = distance
        self.child = child
        self.parent = parent


class OneToAllPathSearchResults:
    """
    Represents results of distance/path finding algorithm (e.g. dijkstra)
    """

    def __init__(self, source: int, lst: list):
        """
        Initialize by a list of PathSearchNode objects
        """
        self.source = source
        self.lst = lst

    def __getitem__(self, index: int) -> PathSearchNode:
        return self.lst[index]

    def __len__(self):
        return len(self.lst)

    def distances(self) -> list:
        """
        Return:
            A list of distances to each node
        """
        return [self[i].distance for i in range(len(self))]

    def children(self) -> list:
        """
        Return:
            A list of children nodes (next nodes in path)
        """
        return [self[i].parent for i in range(len(self))]

    def parents(self) -> list:
        """
        Return:
            A list of parent nodes
        """
        return [self[i].parent for i in range(len(self))]


class AllToAllPathSearchResults(list):
    """
    Represents results of all-to-all searching algorithm
    """

    def __init__(self, matrix: list):
        """
        Args:
            matrix - matrix list of OneToAllPathSearchResults objects
        """
        super().__init__(matrix)


def backtrace_path(search_results: OneToAllPathSearchResults, t: int) -> list:
    """
    Return path from s to t from given precomputed path search results
    """
    path = [t]
    while search_results[t].parent not in [None, undefined_node]:
        assert search_results[t].parent != undefined_node, \
            "Undefined parent while backtracing path"
        t = search_results[t].parent
        path.append(t)
    if path[-1] == search_results.source:
        return list(reversed(path))
    else:
        return None


def forwardtrace_path_from_all_to_all(
        search_results: AllToAllPathSearchResults, s: int, t: int) -> list:
    path = [s]
    while s != t:
        s = search_results[s][t].child
        assert s not in [None, undefined_node]
        path.append(s)
    return path


def distance_matrix(search_results: AllToAllPathSearchResults) -> list:
    matrix = []
    for node_list in search_results:
        matrix.append([node.distance for node in node_list])
    return matrix


def dijkstra(G: WeightedGraph, s: int) -> OneToAllPathSearchResults:
    """
    Args:
        G - graph we search distances in
        s - number (index) of source vertex in G
    Return:
        PathSearchResults object
        None if G has negative edges
    """
    if G.has_negative():
        return None
    found = [False] * G.V()
    parent = [undefined_node] * G.V()
    parent[s] = None
    dist = [sys.maxsize] * G.V()
    dist[s] = 0
    heap = [(0, s)]  # heap of tuples of dist. and vert. number.
    while heap:
        (d, v) = heapq.heappop(heap)
        if found[v]:
            continue
        found[v] = True
        for e in G.adj(v):
            alternate_d = d + e.weight
            if alternate_d < dist[e.dest]:  # relaxation step
                dist[e.dest] = alternate_d
                parent[e.dest] = v
                heapq.heappush(heap, (dist[e.dest], e.dest))

    for i, p in enumerate(parent):
        if p == undefined_node and i != s:
            dist[i] = None

    nodes_list = [PathSearchNode(d, undefined_node, p) for d, p in zip(dist,
                                                                       parent)]
    return OneToAllPathSearchResults(s, nodes_list)


def bellman_ford(G: WeightedDirectedGraph, s: int) -> OneToAllPathSearchResults:
    """
    Args:
        G - graph we search distances in
        s - number (index) of source vertex in G
    Return:
        PathSearchResults object
    """
    parent = [undefined_node] * G.V()
    parent[s] = None
    dist = [sys.maxsize] * G.V()
    dist[s] = 0
    for i in range(G.V()):
        changed = False
        for e in G.edges():
            if parent[e.source] == undefined_node and e.source != s:
                continue
            alternate_d = dist[e.source] + e.weight
            if alternate_d < dist[e.dest]:  # relaxation step
                changed = True
                parent[e.dest] = e.source
                dist[e.dest] = alternate_d
        if not changed:
            break
    else:  # not breaked - then here is a negative cycle
        return None

    for i, p in enumerate(parent):
        if p == undefined_node and i != s:
            dist[i] = None

    nodes_list = [PathSearchNode(d, undefined_node, p) for d, p in zip(dist,
                                                                       parent)]
    return OneToAllPathSearchResults(s, nodes_list)


def floyd_warshall(G: WeightedDirectedGraph) -> AllToAllPathSearchResults:
    """
    Args:
        G - graph to perform search in
    """
    # basic init
    result = AllToAllPathSearchResults([])
    for i in range(G.V()):
        nodes_list = []
        for j in range(G.V()):
            d = 0 if i == j else sys.maxsize
            next_node = None if i == j else undefined_node
            # noinspection PyTypeChecker
            nodes_list.append(PathSearchNode(d, next_node, undefined_node))
        result.append(OneToAllPathSearchResults(i, nodes_list))

    # init as adjacency matrix
    for e in G.edges():
        r = result[e.source][e.dest]
        if r.distance > e.weight:
            r.distance = e.weight
            r.child = e.dest

    # algo
    for k in range(G.V()):  # intermediate vertex
        for i in range(G.V()):  # source vertex
            for j in range(G.V()):  # destination vertex
                # shortcuts
                rij = result[i][j]
                rik = result[i][k]
                rkj = result[k][j]
                if rik.distance == sys.maxsize or rkj.distance == sys.maxsize:
                    continue
                if rij.distance > rik.distance + rkj.distance:
                    # relaxation step
                    rij.distance = rik.distance + rkj.distance
                    rij.child = result[i][k].child

    for i in range(G.V()):
        if result[i][i].distance < 0:  # negative cycle
                return None

    for i, row in enumerate(result):
        for j, res in enumerate(row):
            if res.distance == sys.maxsize:
                res.distance = None

    return result

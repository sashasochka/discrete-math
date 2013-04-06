"""
Different graph algorithm implementations
Works only with zero-based graphs
"""
import sys
import heapq
from graph.weighted.directed import Graph


class PathSearchNode:
    """
    Represents a node for distance searching
    """
    def __init__(self, distance: int, parent: int):
        """
        Initialize public fields from arguments
        """
        self.distance = distance
        self.parent = parent


class PathSearchResults:
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

    def parents(self) -> list:
        """
        Return:
            A list of distances to each node
        """
        return [self[i].parent for i in range(len(self))]


def backtrace_path(search_results: PathSearchResults, t: int) -> list:
    """
    Return path from s to t from given precomputed path search results
    """
    path = [t]
    while search_results[t].parent is not None:
        t = search_results[t].parent
        path.append(t)
    if path[-1] == search_results.source:
        return list(reversed(path))
    else:
        return None


def dijkstra(G: Graph, s: int) -> list:
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
    parent = [None] * G.V()
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
        if p is None and i != s:
            dist[i] = None

    nodes_list = [PathSearchNode(d, p) for d, p in zip(dist, parent)]
    return PathSearchResults(s, nodes_list)


def bellman_ford(G: Graph, s: int) -> list:
    """
    Args:
        G - graph we search distances in
        s - number (index) of source vertex in G
    Return:
        PathSearchResults object
    """
    parent = [None] * G.V()
    dist = [sys.maxsize] * G.V()
    dist[s] = 0
    for i in range(G.V()):
        changed = False
        for e in G.edges():
            if parent[e.source] is None and e.source != s:
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
        if p is None and i != s:
            dist[i] = None

    nodes_list = [PathSearchNode(d, p) for d, p in zip(dist, parent)]
    return PathSearchResults(s, nodes_list)


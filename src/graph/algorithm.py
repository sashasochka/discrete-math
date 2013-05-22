"""
Different graph algorithm implementations
Works only with zero-based graphs
"""
from collections import deque
from operator import itemgetter
import sys
import heapq
from copy import deepcopy, copy
from graph import Graph, Edge
from graph.directed import Graph as DirectedGraph
from graph.directed import Edge as DirectedEdge
from graph.directed import GraphMatrix as DirectedGraphMatrix
from graph.weighted import Graph as WeightedGraph
from graph.weighted import GraphMatrix as WeightedGraphMatrix
from graph.weighted.directed import Graph as WeightedDirectedGraph
from graph.weighted.directed import Edge as WeightedDirectedEdge

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

    def __iter__(self):
        return self.lst.__iter__()

    def __len__(self) -> int:
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
    while search_results[s][t].child not in [None, undefined_node]:
        s = search_results[s][t].child
        path.append(s)
    if path[-1] == t:
        return path
    else:
        return None


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


def johnson(G: WeightedDirectedGraph) -> AllToAllPathSearchResults:
    """
    Perform all-to-all graph search
    Args:
        G - graph to perform search in
    Return:
        Results of search in AllToAllPathSearchResults with filled parent field
    """
    # 1) construct new graph with additional vertex (
    Gq = deepcopy(G)
    q = Gq.add_vertex()  # new vertex number
    for i in range(G.V()):
        Gq.add_edge(WeightedDirectedEdge(q, i, 0))

    # 2) compute distances from q (new vertex)
    bellman_results = bellman_ford(Gq, q)
    if bellman_results is None:  # negative cycle
        return None
    h = bellman_results.distances()[:-1]
    del Gq

    # 3) Create modificated graph without negative edges
    edges_mod = []
    for e in G.edges():
        new_weight = e.weight + h[e.source] - h[e.dest]
        edges_mod.append(WeightedDirectedEdge(e.source, e.dest, new_weight))
    G_mod = WeightedDirectedGraph(G.V(), G.E(), edges_mod)

    # 4) Perform dijkstra searches in this graph
    result = AllToAllPathSearchResults([])
    for i in range(G_mod.V()):
        result.append(dijkstra(G_mod, i))
        for j, r in enumerate(result[-1]):
            if r.distance is not None:
                r.distance -= h[i] - h[j]

    return result


def flatten(lst: list) -> list:
    res = []
    for val in lst:
        if hasattr(val, '__len__'):
            res.extend(flatten(val))
        else:
            res.append(val)
    return res


def eulerian_cycle(G: DirectedGraphMatrix) -> list:
    """
    Find eulerian cycle in the given graph
    Args:
        graph to find eulerian cycle in
    Return:
        path of the cycle if found or None if not found
    """

    def selfloop(u: int):
        nonlocal G
        path = []
        cur = u
        while cur != u or not path:
            path.append(cur)
            (v, c) = G.adj(cur).__next__()
            G.remove_edge(DirectedEdge(cur, v))
            cur = v
        path.append(u)
        return path

    G = deepcopy(G)
    E = G.E()
    tour = [0]
    cur = 0
    while cur < len(tour):
        try:
            G.adj(tour[cur]).__next__()
            tour = tour[:cur] + selfloop(tour[cur]) + tour[cur + 1:]
        except StopIteration:
            cur += 1
    return tour if len(tour) == E + 1 else None


def eulerian_path(G: DirectedGraph) -> list:
    """
    Find eulerian path in the given graph
    Args:
        graph to find eulerian path in
    Return:
        path if found or None if not found
    """

    cycle = eulerian_cycle(G)
    if cycle is not None:
        return cycle

    in_edges = [0] * G.V()
    out_edges = [0] * G.V()
    for u in range(G.V()):
        for v, c in G.adj(u):
            in_edges[v] += c
            out_edges[u] += c

    # start vertex
    s = -1
    cnt = 0
    for v in range(G.V()):
        if (in_edges[v] + out_edges[v]) % 2 == 1:
            cnt += 1
            if cnt > 2:
                return None

            if s == -1 or out_edges[s] % 2 == 0:
                s = v
    if s == -1:
        return None

    G = deepcopy(G)
    E = G.E()

    tour = []
    cur = s
    while True:
        try:
            tour.append(cur)
            (v, c) = G.adj(cur).__next__()
            G.remove_edge(DirectedEdge(cur, v))
            cur = v
        except StopIteration:
            break

    def selfloop(u: int):
        nonlocal G
        path = []
        cur = u
        while cur != u or not path:
            path.append(cur)
            (v, c) = G.adj(cur).__next__()
            G.remove_edge(DirectedEdge(cur, v))
            cur = v
        path.append(u)
        return path

    index = 0
    while index < len(tour):
        try:
            G.adj(tour[index]).__next__()
            tour = tour[:index] + selfloop(tour[index]) + tour[index + 1:]
        except StopIteration:
            index += 1
    return tour if len(tour) == E + 1 else None


def hamiltonian_cycle(G: DirectedGraph) -> list:
    """
    Find hamiltonian cycle in the given graph
    Args:
        graph to find hamiltonian cycle in
    Return:
        path of the cycle if found or None if not found
    """
    if not G.V():
        return []
    if G.V() == 1:
        return [0]

    def dfs(last: int) -> list:
        nonlocal visited, s
        if len(visited) == G.V():
            return [s] if s in G.adj(last) else None

        for v in G.adj(last):
            if v not in visited:
                visited.add(v)
                reversed_path = dfs(v)
                if reversed_path is not None:
                    reversed_path.append(v)
                    return reversed_path
                visited.remove(v)
        return None

    s = 0
    visited = {s}
    result = dfs(s)
    if result is not None:
        result.append(s)
        result = list(reversed(result))
        return result
    else:
        return None


def hamiltonian_path(G: DirectedGraph) -> list:
    """
    Find hamiltonian path in the given graph
    Args:
        graph to find hamiltonian path in
    Return:
        path if found or None if not found
    """
    cycle = hamiltonian_cycle(G)
    if cycle is not None:
        return cycle

    def dfs(last: int) -> list:
        nonlocal visited, s
        if len(visited) == G.V():
            return []

        for v in G.adj(last):
            if v not in visited:
                visited.add(v)
                reversed_path = dfs(v)
                if reversed_path is not None:
                    reversed_path.append(v)
                    return reversed_path
                visited.remove(v)
        return None

    for s in range(G.V()):
        visited = {s}
        result = dfs(s)
        if result is not None:
            result.append(s)
            result = list(reversed(result))
            return result
    return None


def TSP(G: WeightedGraphMatrix):
    if not G.V():
        return 0, []

    prev_dp = {frozenset([0]): [(0, [0])] + [(sys.maxsize, [])] * (G.V() - 1)}
    for _ in range(1, G.V()):
        dp = {}
        # S - set of already visited
        # u - new next visited
        # v - last visited
        for S_prev, vals in prev_dp.items():
            for v in range(G.V()):
                dist = vals[v][0]
                if dist >= sys.maxsize:
                    continue
                for u in range(G.V()):
                    d = G.distance[v][u]
                    if u not in S_prev and d is not None:
                        S = set(S_prev)
                        S.add(u)
                        S = frozenset(S)

                        if S in dp:
                            new_vals = dp[S]
                        else:
                            new_dists = [sys.maxsize] * G.V()
                            new_paths = [[] for _ in range(G.V())]
                            new_vals = list(zip(new_dists, new_paths))

                        if dist + d < new_vals[u][0]:
                            new_vals[u] = (dist + d, vals[v][1] + [u])
                        dp[S] = new_vals
        prev_dp = dp
    best = (sys.maxsize, [])
    for i, (dst, path) in enumerate(prev_dp[frozenset(range(G.V()))]):
        if G.distance[i][0] is not None:
            altern_dst = dst + G.distance[i][0]
            if altern_dst < best[0]:
                best = (altern_dst, path)
    best[1].append(best[1][0])
    return best


def isK33(G: Graph):
    edges = []
    for u in range(3):
        for v in range(3, 6):
            edges.append(Edge(u, v))
    return G == Graph(6, len(edges), edges)


def isK5(G: Graph):
    edges = []
    for u in range(5):
        for v in range(u + 1, 5):
            edges.append(Edge(u, v))
    return G == Graph(5, len(edges), edges)


def is_planar(G: Graph) -> bool:
    """
    Returns true if graph G is planar

    (horrible exponential implementation)
    FIXME: rewrite using linear algorithm

    Examples:
    >>> is_planar(Graph(1,0,[]))
    True
    >>> is_planar(Graph.fromfile(open('K33.txt')))
    False
    >>> is_planar(Graph.fromfile(open('K4.txt')))
    True
    >>> is_planar(Graph.fromfile(open('K5.txt')))
    False
    >>> is_planar(Graph.fromfile(open('petersen.txt')))
    False
    """
    for mask in range(1 << G.E()):
        remove_edges = []
        left_edges = []
        for i, e in enumerate(G.edges()):
            if (mask >> i) % 2 == 1:
                remove_edges.append(tuple(sorted([e.u, e.v])))
            else:
                left_edges.append(copy(e))
        remove_edges.sort(key=itemgetter(1), reverse=True)
        for u, v in remove_edges:
            for e in left_edges:
                if e.v == v:
                    e.v = u
                elif e.v > v:
                    e.v -= 1
                if e.u == v:
                    e.u = u
                elif e.u > v:
                    e.u -= 1
        left_edges = sorted(left_edges)
        left_edg = []
        st = None
        V = -1
        for e in left_edges:
            V = max(V, e.u + 1, e.v + 1)
            if e != st and e.u != e.v:
                left_edg.append(e)
            else:
                st = e
        contractedG = Graph(V, len(left_edg), left_edg)
        if isK5(contractedG) or isK33(contractedG):
            return False
    return True


def coloring(G: Graph):
    colors = []

    def color(k: int) -> bool:
        nonlocal colors
        for mask in range(k ** G.V()):
            colors = [(mask // k ** i) % k for i in range(G.V() - 1, -1, -1)]
            for e in G.edges():
                if colors[e.u] == colors[e.v]:
                    break
            else:
                return True
        return False

    for k in range(G.V() + 1):
        if color(k):
            return k, [i + 1 for i in colors]


class ambigous_source(Exception):
    pass


class no_source(Exception):
    pass


class ambigous_sink(Exception):
    pass


class no_sink(Exception):
    pass


def get_source(G: WeightedDirectedGraph) -> int:
    """
    Return index of source vertex in graph G
    Example:
    >>> get_source(WeightedDirectedGraph.fromfile(open('wiki_flow.txt')))
    0
    >>> get_source(WeightedDirectedGraph(0,0,[]))
    Traceback (most recent call last):
        ...
    algorithm.no_source
    >>> get_source(WeightedDirectedGraph(2,0,[]))
    Traceback (most recent call last):
        ...
    algorithm.ambigous_source
    """
    in_degree = [0] * G.V()
    for e in G.edges():
        in_degree[e.dest] += 1
    V = [v for v in range(G.V()) if in_degree[v] == 0]
    if not V:
        raise no_source
    if len(V) > 1:
        raise ambigous_source
    return V[0]


def get_sink(G: WeightedDirectedGraph) -> int:
    """
    Return index of output vertex in graph G
    Example:
    >>> get_sink(WeightedDirectedGraph.fromfile(open('wiki_flow.txt')))
    5
    >>> get_sink(WeightedDirectedGraph(0,0,[]))
    Traceback (most recent call last):
        ...
    algorithm.no_sink
    >>> get_sink(WeightedDirectedGraph(3,2,
    ... [WeightedDirectedEdge(0, 1, 1), WeightedDirectedEdge(0, 2, 1)]))
    Traceback (most recent call last):
        ...
    algorithm.ambigous_sink
    """
    V = [v for v in range(G.V()) if len(G.adj(v)) == 0]
    if not V:
        raise no_sink
    if len(V) > 1:
        raise ambigous_sink
    return V[0]


def edmonds_karp(G: WeightedDirectedGraph, s: int, t: int) -> tuple:
    """
    Find maximal flow in graph from source s to sink t
    Return tuple of (weight: int, edges_with_weights)
    edges_with_weights is list of tuples of (edge_weight: int, edge: Edge)
    Example:
    """
    def bfs():
        parent = [None] * G.V()
        parent[s] = s
        q = deque([])
        q.append(s)
        while q:
            current = q.popleft()
            for adjacent in adjacent_to[current]:
                has_parent = parent[adjacent] is not None
                left_flow_on_edge = M[current][adjacent] > F[current][adjacent]
                can_go_backward = F[adjacent][current] > 0
                if not has_parent and (left_flow_on_edge or can_go_backward):
                    parent[adjacent] = current
                    q.append(adjacent)
                    if adjacent == t:
                        return parent
        return parent

    def path_flow(parent):
        result = sys.maxsize
        cur = t
        while cur != s:
            # from parent[cur] to cur
            if M[parent[cur]][cur] > F[parent[cur]][cur]:
                flow_here = M[parent[cur]][cur] - F[parent[cur]][cur]
            else:
                flow_here = F[cur][parent[cur]]
            cur = parent[cur]
            result = min(result, flow_here)
        return result

    def add_flow_on_path(parent, val):
        cur = t
        while cur != s:
            F[parent[cur]][cur] += val
            F[cur][parent[cur]] -= val
            cur = parent[cur]

    flow_value = 0
    F = [[0] * G.V() for _ in range(G.V())]
    M = [[0] * G.V() for _ in range(G.V())]
    adjacent_to = [[] for _ in range(G.V())]
    for edge in G.edges():
        M[edge.source][edge.dest] = edge.weight
        adjacent_to[edge.source].append(edge.dest)
        adjacent_to[edge.dest].append(edge.source)

    while True:
        parent = bfs()
        if parent[t] is not None:
            path_flow_val = path_flow(parent)
            flow_value += path_flow_val
            add_flow_on_path(parent, path_flow_val)
        else:
            flow_edges = []
            for source in range(G.V()):
                for dest, flow in enumerate(F[source]):
                    if flow > 0:
                        flow_edges.append(
                            WeightedDirectedEdge(source, dest, flow)
                        )
            return flow_value, \
                WeightedDirectedGraph(G.V(), len(flow_edges), flow_edges)

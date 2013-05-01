"""
Base graph classes (unweigheted and undirected)
"""
import sys


class Edge:
    """
    Represent abstract edge
    """

    def __init__(self, u: int, v: int):
        self.u = u
        self.v = v

    def either(self) -> int:
        """
        Return 1st vertex number
        """
        return self.u

    def other(self) -> int:
        """
        Return 2nd vertex number
        """
        return self.v

    def __lt__(self, other) -> bool:
        return sorted([self.u, self.v]) < sorted([other.u, other.v])

    def __eq__(self, other) -> bool:
        if not hasattr(other, 'u') or not hasattr(other, 'v'):
            return False
        return sorted([self.u, self.v]) == sorted([other.u, other.v])

    def __str__(self):
        return '{} {}'.format(self.u, self.v)


class Graph:
    """
    Basic graph (undweighted and undirected)
    """

    def __init__(self, V: int, E: int, edges: list):
        """
        Args:
            V is the number of vertexes
            E is the number of edges
            edges is the list of Edge objects
        """
        self._V = V
        self._E = E
        self._edges = edges
        self._adj = [[] for _ in range(V)]
        for e in edges:
            self._adj[e.u].append(e.v)
            self._adj[e.v].append(e.u)

    @classmethod
    def fromfile(cls, readobj: type(sys.stdin), one_indexation: bool=True):
        """
        Initialize object from readable file
        Args:
            readobj - readable object with input data in correcponding format
        Return:
            correctly initialized Graph object
        """
        V, E = map(int, readobj.readline().split())
        edges = []
        for i in range(E):
            line = readobj.readline()
            u, v = map(int, line.split())
            if one_indexation:
                u -= 1
                v -= 1
            edges.append(Edge(u, v))
        return cls(V, E, edges)

    def V(self) -> int:
        """
        Return:
            number of vertexes
        """
        return self._V

    def E(self) -> int:
        """
        Return:
            number of edges
        """
        return self._E

    def edges(self) -> list:
        """
        Return:
            list of edges in graph
        """
        return self._edges

    def adj(self, source: int) -> list:
        """
        Args:
            source - vertex number
        Return:
            list of all edges incident to source
        """
        assert 0 <= source < self._V
        return self._adj[source]

    def add_vertex(self) -> int:
        self._V += 1
        self._adj.append([])
        return self._V - 1

    def add_edge(self, edge: Edge):
        raise NotImplementedError

    def __str__(self) -> str:
        result = '{} {}\n{}'.format(
            self._V,
            self._E,
            '\n'.join(map(str, self.edges())))
        return result

    def __eq__(self, other) -> bool:
        return self.E() == other.E() and self.V() == other.V() and (
            sorted(self.edges()) == sorted(other.edges()))

    def __lt__(self, other) -> bool:
        return tuple(sorted([self.u, self.v])) < \
            tuple(sorted([other.u, other.v]))


class GraphMatrix:
    """
    Same as Graph but using adjacency matrix and different API
    """

    def __init__(self, V: int, E: int, edges: list):
        """
        Args:
            V is the number of vertexes
            E is the number of edges
            edges is the list of Edge objects
        """
        self._V = V
        self._E = E
        self.paths = [[0] * V for _ in range(V)]
        for e in edges:
            self.paths[e.u][e.v] += 1
            self.paths[e.v][e.u] += 1

    @classmethod
    def fromfile(cls, readobj: type(sys.stdin), one_indexation: bool=True):
        """
        Initialize object from readable file
        Args:
            readobj - readable object with input data in correcponding format
        Return:
            correctly initialized Graph object
        """
        V, E = map(int, readobj.readline().split())
        edges = []
        for i in range(E):
            line = readobj.readline()
            u, v = map(int, line.split())
            if one_indexation:
                u -= 1
                v -= 1
            edges.append(Edge(u, v))
        return cls(V, E, edges)

    @classmethod
    def from_graph(cls, graph: Graph):
        """
        Initialize graph using another graph which is in adjacency list
        representation
        """
        return cls(graph.V(), graph.E(), graph.edges())

    def V(self):
        return self._V

    def E(self):
        return self._E

    def add_edge(self, edge: Edge):
        assert 0 <= edge.u < self.V()
        assert 0 <= edge.v < self.V()
        self.paths[edge.u][edge.v] += 1
        self.paths[edge.v][edge.u] += 1

    def remove_edge(self, edge: Edge):
        assert self.paths[edge.u][edge.v] != 0
        self.paths[edge.u][edge.v] -= 1
        self.paths[edge.v][edge.u] -= 1

    def adj(self, u):
        assert 0 <= u < self.V()
        for v in range(self.V()):
            if self.paths[u][v]:
                yield Edge(u, v)

    def adjacent_to(self, u):
        return sum(self.paths[u])

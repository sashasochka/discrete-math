"""
Includes directed graph classes
"""
from itertools import chain
import sys
import graph


class Edge(graph.Edge):
    """
    Represent directed edge
    """
    def __init__(self, u: int, v: int):
        """
        Args:
            u - first vertex
            v- isecond vertex
        """
        self.u = u
        self.v = v

    def reverse(self) -> 'Edge':
        """
        Return:
            edge with reversed direction
        """
        return Edge(self.v, self.u)

    def either(self):
        return self.u

    def other(self):
        return self.v

    def __str__(self):
        return '{} {}'.format(self.u, self.v)


class Graph(graph.Graph):
    """
    Represents Directed Graph (not weighted)
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

    def add_edge(self, edge: Edge):
        assert 0 <= edge.u < self.V()
        assert 0 <= edge.v < self.V()
        self._edges.append(edge)
        self._adj[edge.u].append(edge.v)

    def reverse(self) -> 'Graph':
        return Graph(
            self.V(),
            self.E(),
            [e.reverse() for e in self.edges()]
        )


class GraphMatrix():
    """
    Represents Directed Graph based on adjacency matrix (not weighted)
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

    def remove_edge(self, edge: Edge):
        assert self.paths[edge.u][edge.v] != 0
        self.paths[edge.u][edge.v] -= 1

    def adj(self, u):
        assert 0 <= u < self.V()
        for v in range(self.V()):
            if self.paths[u][v]:
                yield (v, self.paths[u][v])

    def adjacent_to(self, u):
        return sum(self.paths[u])

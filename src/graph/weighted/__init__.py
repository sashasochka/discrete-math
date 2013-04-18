"""
Classes for weighted graph
"""
import sys
import graph


class Edge(graph.Edge):
    """
    Represent directed weighted edge
    """
    def __init__(self, u: int, v: int, weight: float):
        """
        Args:
            source - is a source vertex
            dest - is a destination vertex
            weight - is a weight of this edge
        """
        self.u = u
        self.v = v
        self.weight = weight

    def reverse(self) -> 'Edge':
        """
        Return:
            edge with reversed direction
        """
        return Edge(self.v, self.u, self.weight)

    def either(self):
        return self.u

    def other(self):
        return self.v

    def negative(self) -> bool:
        """
        Return:
            True if weight is negative
            else False
        """
        return self.weight < 0

    def __str__(self) -> str:
        """
        Return:
            string representation
        """
        return '{} {} {}'.format(self.either(), self.other(), self.weight)


class Graph(graph.Graph):
    """
    Represents Directed Weighted Graph
    """

    def __init__(self):
        """
        Args:
            V is the number of vertexes
            E is the number of edges
            edges is the list of Edge objects
        """
        raise NotImplementedError

    def weight(self) -> int:
        return sum([e.weight for e in self.edges()])

    def has_negative(self) -> bool:
        """
        Return:
            True if graph has negative edges
            else False
        """
        return self._negative


class GraphMatrix():
    """
    Represents Weighted Directed Graph based on adjacency matrix (not weighted)
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
        self.distance = [[None] * V for _ in range(V)]
        for e in edges:
            self.distance[e.u][e.v] = e.weight
            self.distance[e.v][e.u] = e.weight

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
            u, v, weight = map(int, line.split())
            if one_indexation:
                u -= 1
                v -= 1
            edges.append(Edge(u, v, weight))
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
        self.distance[edge.u][edge.v] = edge.weight
        self.distance[edge.v][edge.u] = edge.weight

    def remove_edge(self, edge: Edge):
        assert self.distance[edge.u][edge.v] != 0
        self.distance[edge.u][edge.v] -= 1

    def adj(self, u):
        assert 0 <= u < self.V()
        for v in range(self.V()):
            if self.distance[u][v]:
                yield (v, self.distance[u][v])

    def adjacent_to(self, u):
        return sum(1 if self.distance[u] else 0)

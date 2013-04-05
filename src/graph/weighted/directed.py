"""
Includes weighted directed graph classes
"""
import itertools


class Graph:
    """
    Represents Directed Weighted Graph
    """

    def __init__(self, E, V, edges):
        """
        (int, int, list of Edge) -> void
        Args:
            E is the number of vertexes
            V is the number of edges
            edges is the list of Edge objects
        """
        self.__E = E
        self.__V = V
        self.__edges = edges

    @classmethod
    def fromfile(cls, readobj):
        """
        (file) -> Graph
        Initialize object from readable file
        Args:
            readobj - readable object with input data in correcponding format
        Return:
            correctly initialized Graph object
        """
        V, E = map(int, readobj.readline().split())
        edges = []
        for line in readobj:
            source, dest, width = map(int, readobj.readline().split())
            edges.append(Edge(source, dest, width))
        return cls(V, E, edges)

    def V(self):
        """
        () -> int
        Return:
            number of vertexes
        """
        return self.__V

    def E(self):
        """
        () -> int
        Return:
            number of edges
        """
        return self.__E

    def edges(self):
        """
        () -> list of Edge
        Return:
            full list of edges in graph
        """
        return itertools.chain(*self.__edges)

    def adj(self, source):
        """
        (int) -> list of Edge
        Args:
            source - vertex number
        Return:
            list of all edges from source
        """
        assert 0 <= source < self.V()
        return self.__edges[source]


class Edge:
    """
    Represent directed weighted edge
    """
    def __init__(self, source, dest, weight):
        """
        (int, int, int) -> void
        Args:
            source - is a source vertex
            dest - is a destination vertex
            weight - is a weight of this edge
        """
        self.source = source
        self.dest = dest
        self.weight = weight

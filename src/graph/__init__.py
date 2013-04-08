"""
Abstract graph classes
"""
import sys


class Edge:
    """
    Represent abstract edge
    """

    def to_zero_based(self) -> 'Edge':
        """
        Change to 0-based index. Call only if 1-based now!
        Return:
            edge with decreased vertexes numbers by one
        """
        raise NotImplementedError

    def to_one_based(self) -> 'Edge':
        """
        Change to 1-based index. Call only if 0-based now!
        Return:
            edge with increased vertexes numbers by one
        """
        raise NotImplementedError

    def either(self) -> int:
        """
        Return 1st vertex number
        """
        raise NotImplementedError

    def other(self) -> int:
        """
        Return 2nd vertex number
        """


class Graph:
    """
    Represents abstract graph
    """
    def __init__(self):
        """
        Args:
            V is the number of vertexes
            E is the number of edges
            edges is the list of Edge objects
        """
        self._adj = []
        self._edges = []
        self._negative = False
        self._V = 0
        self._E = 0
        raise NotImplementedError

    @classmethod
    def fromfile(cls, readobj: type(sys.stdin), one_indexation: bool=True):
        """
        Initialize object from readable file
        Args:
            readobj - readable object with input data in correcponding format
        Return:
            correctly initialized Graph object
        """
        raise NotImplementedError

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

    def has_negative(self) -> bool:
        """
        Return:
            True if graph has negative edges
            else False
        """
        return self._negative

    def add_vertex(self):
        self._V += 1
        self._adj.append([])

    def add_edge(self, edge: Edge):
        raise NotImplementedError

    def __str__(self) -> str:
        result = '{}\n{}\n{}'.format(
            self._V,
            self._E,
            '\n'.join(map(str, self.edges())))
        return result

"""
Includes weighted directed graph classes
"""
import itertools
import sys


class Graph:
    """
    Represents Directed Weighted Graph
    """

    def __init__(self, V: int, E: int, edges: list):
        """
        Args:
            V is the number of vertexes
            E is the number of edges
            edges is the list of Edge objects
        """
        self.__V = V
        self.__E = E
        self.__adj = [[] for _ in range(V)]
        self.__negative = False
        for e in edges:
            self.__adj[e.source].append(e)
            if e.negative():
                self.__negative = True

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
        for line in readobj:
            source, dest, width = map(int, line.split())
            if one_indexation:
                source -= 1
                dest -= 1
            edges.append(Edge(source, dest, width))
        return cls(V, E, edges)

    def V(self) -> int:
        """
        Return:
            number of vertexes
        """
        return self.__V

    def E(self) -> int:
        """
        Return:
            number of edges
        """
        return self.__E

    def edges(self) -> list:
        """
        Return:
            list of edges in graph
        """
        return list(itertools.chain(*self.__adj))

    def adj(self, source: int) -> list:
        """
        Args:
            source - vertex number
        Return:
            list of all edges incident to source
        """
        assert 0 <= source < self.V()
        return self.__adj[source]

    def has_negative(self) -> bool:
        """
        Return:
            True if graph has negative edges
            else False
        """
        return self.__negative

    def reverse(self) -> 'Graph':
        return Graph(
            self.V(),
            self.E(),
            [e.reverse() for e in self.edges()]
        )

    def __str__(self) -> str:
        result = '{}\n{}\n{}'.format(
            self.V(),
            self.E(),
            '\n'.join(map(str, self.edges())))
        return result


class Edge:
    """
    Represent directed weighted edge
    """
    def __init__(self, source: int, dest: int, weight: int):
        """
        Args:
            source - is a source vertex
            dest - is a destination vertex
            weight - is a weight of this edge
        """
        self.source = source
        self.dest = dest
        self.weight = weight

    def reverse(self) -> 'Edge':
        """
        Return:
            edge with reversed direction
        """
        return Edge(self.dest, self.source, self.weight)

    def negative(self) -> bool:
        """
        Return:
            True if weight is negative
            else False
        """
        return self.weight < 0

    def to_zero_based(self) -> 'Edge':
        """
        Change to 0-based index. Call only if 1-based now!
        Return:
            edge with decreased vertexes numbers by one
        """
        return Edge(self.source - 1, self.dest - 1, self.weight)

    def to_one_based(self) -> 'Edge':
        """
        Change to 1-based index. Call only if 0-based now!
        Return:
            edge with increased vertexes numbers by one
        """
        return Edge(self.source + 1, self.dest + 1, self.weight)

    def __str__(self) -> str:
        """
        Return:
            string representation
        """
        return '{} {} {}'.format(self.source, self.dest, self.weight)


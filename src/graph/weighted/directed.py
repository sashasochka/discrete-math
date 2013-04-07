"""
Includes weighted directed graph classes
"""
import sys
import graph.weighted


class Graph(graph.weighted.Graph):
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
        self._V = V
        self._E = E
        self._edges = edges
        self._adj = [[] for _ in range(V)]
        self._negative = False
        for e in edges:
            self._adj[e.source].append(e)
            if e.negative():
                self._negative = True

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

    def reverse(self) -> 'Graph':
        return Graph(
            self.V(),
            self.E(),
            [e.reverse() for e in self.edges()]
        )


class Edge(graph.weighted.Edge):
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

    def either(self):
        return self.source

    def othre(self):
        return self.dest


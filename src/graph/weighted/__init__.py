"""
Abstract classes for weighted graph
"""
import graph


class Edge(graph.Edge):
    """
    Represent directed weighted edge
    """
    def __init__(self):
        """
        Args:
            source - is a source vertex
            dest - is a destination vertex
            weight - is a weight of this edge
        """
        self.weight = 0
        raise NotImplementedError

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

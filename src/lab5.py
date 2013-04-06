#!/usr/bin/env python
from common import IO, UserIO
from graph import algorithm
from graph.weighted.directed import Graph


io = IO()
userio = UserIO()
G = Graph.fromfile(io.filein)

s = userio.readvertex(G.V(), 'source')
t = userio.readvertex(G.V(), 'destination')
if G.has_negative():
    userio.print('Cannot find distance using dijkstra - negative edges are '
                 'present')
else:
    # print distances from s to other vertexes
    search_results = algorithm.dijkstra(G, s)
    userio.print('Distances from source vertex (#{}) using dijkstra\'s '
                 'algorithm: '.format(s + 1))
    userio.print_numbered_list(search_results.distances(), 'Distance')

    # print path from s to t
    path = algorithm.backtrace_path(search_results, t)
    if path is None:
        userio.print('There is no path from vertex #{} to #{}'.format(s + 1,
                                                                      t + 1))
    else:
        userio.print('Path from #{} to #{}:'.format(s + 1, t + 1))
        userio.print(' -> '.join([str(i + 1) for i in path]))


# TODO: add bellman-ford

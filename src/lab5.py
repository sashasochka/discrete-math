#!/usr/bin/env python3
from common import IO
from graph import algorithm
from graph.weighted.directed import Graph


io = IO()
G = Graph.fromfile(io.filein)

s = io.user.readvertex(G.V(), 'source')
t = io.user.readvertex(G.V(), 'destination')

# dijkstra part
io.section('Dijkstra\'s algorithm')
if G.has_negative():
    io.print('Cannot find distance using dijkstra - negative edges are '
             'present')
else:
    # print distances from s to other vertexes
    search_results = algorithm.dijkstra(G, s)
    io.print('Distances from source vertex (#{}) using Dijkstra\'s '
             'algorithm: '.format(s + 1))
    io.print_numbered_list(search_results.distances(), 'Distance')

    # print path from s to t
    path = algorithm.backtrace_path(search_results, t)
    if path is None:
        io.print('There is no path from vertex #{} to #{}'.format(s + 1,
                                                                  t + 1))
    else:
        io.print('Path from #{} to #{}:'.format(s + 1, t + 1))
        io.print(' -> '.join([str(i + 1) for i in path]))

# bellman-ford part
# print distances from s to other vertexes
io.section('Bellman-Ford\'s algorithm')
search_results = algorithm.bellman_ford(G, s)
if search_results is None:
    io.print('Negative cycle present. There is no shortest paths')
else:
    io.print('Distances from source vertex (#{}) using Bellman-Ford\'s '
             'algorithm: '.format(s + 1))
    io.print_numbered_list(search_results.distances(), 'Distance')

    # print path from s to t
    path = algorithm.backtrace_path(search_results, t)
    if path is None:
        io.print('There is no path from vertex #{} to #{}'.format(s + 1,
                                                                  t + 1))
    else:
        io.print('Path from #{} to #{}:'.format(s + 1, t + 1))
        io.print(' -> '.join([str(i + 1) for i in path]))

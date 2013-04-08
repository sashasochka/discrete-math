#!/usr/bin/env python3
from common import IO, UserIO
from graph import algorithm
from graph.weighted.directed import Graph

io = IO()
userio = UserIO()
G = Graph.fromfile(io.filein)

# later defined source and destination vertexes
s = 0
t = 0

# Floyd-Warshall's part
io.section('Floyd-Warshall\'s algorithm')
search_results = algorithm.floyd_warshall(G)
if search_results is None:
    io.print('Cannot find distance using Floyd-Warshall - negative cycles are '
             'present')
else:
    # print distance matrix
    io.print('Distance matrix: ')
    io.print_matrix(algorithm.distance_matrix(search_results), 's\\t')

    # ask user for source and destination vertexes
    s = userio.readvertex(G.V(), 'source')
    t = userio.readvertex(G.V(), 'destination')

    # print path from s to t
    path = algorithm.forwardtrace_path_from_all_to_all(search_results, s,
                                                       t)
    if path is None:
        io.print('There is no path from vertex #{} to #{}'.format(s + 1,
                                                                  t + 1))
    else:
        io.print('Path from #{} to #{}:'.format(s + 1, t + 1))
        io.print(' -> '.join([str(i + 1) for i in path]))

# Johnson's part
io.section('Johnson\'s algorithm')
search_results = algorithm.johnson(G)
if search_results is None:
    io.print('Cannot find distance using Johnson - negative cycles are '
             'present')
else:
    # print distance matrix
    io.print('Distance matrix: ')
    io.print_matrix(algorithm.distance_matrix(search_results), 's\\t')

    # print path from s to t
    path = algorithm.backtrace_path(search_results[s], t)
    if path is None:
        io.print('There is no path from vertex #{} to #{}'.format(s + 1,
                                                                  t + 1))
    else:
        io.print('Path from #{} to #{}:'.format(s + 1, t + 1))
        io.print(' -> '.join([str(i + 1) for i in path]))

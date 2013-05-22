#!/usr/bin/env python3
from common import IO
from graph import algorithm
from graph.weighted.directed import Graph

io = IO()
G = Graph.fromfile(io.filein)

io.section('Max flow algorithm (Floyd-Fulkerson method, '
           'Edmonds-Karp algorithm)')
try:
    s = algorithm.get_source(G)
    t = algorithm.get_sink(G)
    flow_val, flow_graph = algorithm.edmonds_karp(G, s, t)
    io.print('Max flow value is {}\n'.format(flow_val))
    io.print('Flow on each edge: ')
    for edge in flow_graph.edges():
        io.print('\tfor edge ({}, {}) flow is {}'.format(edge.source,
                                                      edge.dest,
                                                      edge.weight))
except algorithm.ambigous_source:
    io.print('Source is ambigous (more than 2 vertexes with out-degree = 0')
except algorithm.no_source:
    io.print('Cannot find source - there are no vertexes with out-degree = 0')
except algorithm.ambigous_sink:
    io.print('Source is ambigous (more than 2 vertexes with in-degree = 0')
except algorithm.no_sink:
    io.print('Cannot find sink - there are no vertexes with in-degree = 0')

#!/usr/bin/env python3
from math import sqrt
from common import IO
from graph.weighted import GraphMatrix, Edge
from graph import algorithm

io = IO()

N = int(io.readline())
coords = []
G = GraphMatrix(N, 0, [])
for i in range(N):
    x, y = map(float, io.readline().split())
    coords.append((x, y))
    for j, (x2, y2) in enumerate(coords):
        dist = sqrt((x2 - x) ** 2 + (y2 - y) ** 2)
        G.add_edge(Edge(i, j, dist))


io.section('TSP path')
weight, path = algorithm.TSP(G)
if path is not None:
    io.print('TSP path with weight {} found:'.format(weight))
    io.print_path(path)
else:
    io.print('TSP path not found')

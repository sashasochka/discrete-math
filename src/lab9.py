#!/usr/bin/env python3
from common import IO
from graph import algorithm, Graph, GraphMatrix

io = IO()
G = Graph.fromfile(io.filein)

# Part 1: graph planatity
io.section('Graph planarity')
if algorithm.is_planar(G):
    io.print('Graph is planar')
else:
    io.print('Graph is not planar!')

# Part 2: graph coloring
io.section('Graph coloring')
k, colors = algorithm.coloring(G)
io.print('Graph can be colored using {} colors, for example like this:'
  .format(k))
io.print_numbered_list(colors)


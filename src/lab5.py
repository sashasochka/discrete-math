#!/usr/bin/env python
import sys
from common import IO
from graph.weighted.directed import Graph


io = IO()
graph = Graph.fromfile(io.filein)

io.print('hello', end='!')

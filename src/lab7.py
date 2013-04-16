#!/usr/bin/env python3
from common import IO
from graph.directed import Graph, GraphMatrix
from graph import algorithm

io = IO()

G = Graph.fromfile(io.filein)
G_matrix = GraphMatrix.from_graph(G)

# Eulerian cycle, part 1
io.section('Eulerian cycle')
eulerian_cycle = algorithm.eulerian_cycle(G_matrix)
if eulerian_cycle is not None:
    io.print('Eulerian cycle found: ')
    io.print_path(eulerian_cycle)
else:
    io.print('Eulerian cycle not found!')

    # Eulerian path, part 2
    io.section('Eulerian path')
    eulerian_path = algorithm.eulerian_path(G_matrix)
    if eulerian_path is not None:
        io.print('Eulerian path found: ')
        io.print_path(eulerian_path)
    else:
        io.print('Eulerian path not found!')


# Hamiltonian cycle, part 1
io.section('Hamiltonian cycle')
hamiltonian_cycle = algorithm.hamiltonian_cycle(G)
if hamiltonian_cycle is not None:
    io.print('Hamiltonian cycle found: ')
    io.print_path(hamiltonian_cycle)
else:
    io.print('Hamiltonian cycle not found!')

    # Hamiltonian path, part 2
    io.section('Hamiltonian path')
    hamiltonian_path = algorithm.hamiltonian_path(G)
    if hamiltonian_path is not None:
        io.print('Hamiltonian path found: ')
        io.print_path(hamiltonian_path)
    else:
        io.print('Hamiltonian path not found!')

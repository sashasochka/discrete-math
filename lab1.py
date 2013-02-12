#!/usr/bin/env python3
import directed_graph as graph
from functions import *
import sys

def main():
  # input
  fin = get_input_file()
  fout = get_output_file()
  if fin == sys.stdin:
    print('Input graph: ')
  
  # part 1
  n, m, E = G = graph.readAsEdgeList(fin)
  
  # part 2
  fout.write('\n\nIncidentness matrix (rows - vertex, columns - edges): \n')
  print_matrix(graph.incidentMatrixByEdgeList(G), fout, 'v\e', 3)

  fout.write('\n\nAdjacency matrix: \n')
  print_matrix(graph.adjacencyMatrixByEdgeList(G), fout, 'from\\to', 3)

  # part 3
  inDegree, outDegree = graph.inOutDegreesByEdgeList(G)
  fout.write('\n\nOut degree per vertex table: \n')
  print_vector(outDegree, fout, 'Out degree', 3)
  fout.write('\n\nIn degree per vertex table: \n')
  print_vector(inDegree,  fout, 'In degree',  3)

  degree = [sum(pair) for pair in zip(inDegree, outDegree)]
  if not any(d != degree[0] for d in degree):
    fout.write(' Graph is uniform, uniformness degree is ' + str(degree[0]) + '\n')
  else:
    fout.write(' Graph is not uniform')

  # part 4
  if ask('Do you want to get all terminal and isolated vertexes? YES/no?'):
    terminal = [i+1 for i in range(n) if degree[i] == 1]
    isolated = [i+1 for i in range(n) if degree[i] == 0]
    fout.write('\n\nTerminal vertexes list: \n')
    print(terminal)
    fout.write('\n\nIsolated vertexes list: \n')
    print(isolated)

main()
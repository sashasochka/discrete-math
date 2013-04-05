#!/usr/bin/env python3
import directed_graph as graph
from functions import *
from functools import reduce
import sys

def main():
  # input
  fin = get_input_file()
  fout = get_output_file()
  if fin == sys.stdin:
    print('Input graph: ')
  
  n, m, E = G = graph.readAsEdgeList(fin)
  
  adj_G = graph.edgeListToAdjacencyList(G)

  # part 1
  if ask('Do you want to get distance and reachability?'):
    D = graph.wavingDistance(adj_G)
    for i in range(n):
      for j in range(n):
        if D[i][j] == graph.inf:
          D[i][j] = '-'

    fout.write('\n\n Distance matrix: \n')
    print_matrix(D, fout, 's\\t', 3)

    R = graph.wavingReachable(adj_G)
    fout.write('\n\n Reachability matrix: \n')
    print_matrix(R, fout, 's\\t', 3)

  # part 2
  loops = graph.get_loops(adj_G)
  if loops:
    fout.write('\n\n Loops are present, some of them: \n')
    for i, loop in enumerate(loops):
      print('Loop {}:'.format(str(i+1)), ' -> '.join([str(i+1) for i in loop]))
  else:
    fout.write('\n\n Graph is loop-free \n')

  # part 3
  if graph.isStronglyConnectedByRMatrix(R):
    fout.write('\n\nGraph is strongly connected\n')
  elif graph.isOneSideConnectedByRMatrix(R):
    fout.write('\n\nGraph is one-side connected\n')
  elif graph.isWeaklyConnectedByEdgeList(G):
    fout.write('\n\nGraph is weakly connected\n')
  else:
    fout.write('\n\nGraph is not connected at all\n')

main()
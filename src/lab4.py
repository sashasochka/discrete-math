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
  
  n, m, E = G = graph.readAsEdgeList(fin)
  adj_G = graph.edgeListToAdjacencyList(G)
  
  # part1
  print('\n')
  top_sort = graph.topological_sort(adj_G)
  if top_sort != None:
    print_vector(top_sort, fout, 'Vertex')
  else:
    print('Graph has cycles!')

  # part2
  print('\n')
  print_vector(graph.strong_components(adj_G), fout, 'Strongly connected components')

if __name__ == '__main__':
  main()
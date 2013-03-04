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
  
  # part0:
  s = int(input('Enter number from {} to {}: '.format(1, n)))
  assert(1 <= s <= n)

  # part1:
  print('\nBFS table:\n')
  print_table(graph.bfs_table(adj_G, s - 1), fout,
    ['BFS-number', 'Vertex number', 'Current queue'], 10)

  # part2:
  print('\nDFS table:\n')
  print_table(graph.dfs_table(adj_G, s - 1), fout,
    ['DFS-number', 'Vertex number', 'Current stack'], 10)


if __name__ == '__main__':
  main()
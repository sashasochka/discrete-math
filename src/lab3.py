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
  
  # adj_G = graph.edgeListToAdjacencyList(G)

if __name__ == '__main__':
  main()
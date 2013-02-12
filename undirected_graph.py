#!/usr/bin/env python3
from sys import stdin
from queue import Queue

def readAsEdgeList(file = stdin):
  ''' unpack graph as (n, m, e) = readAsEdgeList(file) '''
  n, m = map(int, file.readline().split())
  E = [map(int, file.readline().split()) for i in range(m)]
  return n, m, E

def incidentMatrixByEdgeList(G):
  n, m, E = G
  matrix = [[0] * m for i in range(n)]
  for (i, e) in enumerate(E):
    (v1, v2) = e
    matrix[v1 - 1][i] += 1
    matrix[v2 - 1][i] += 1
  return matrix


def edgeListToAdjacencyList(G):
  n, m, E = G
  adj_lst = [[] for i in range(n)]
  for (v1, v2) in E:
    adj_lst[v1-1].append(v2-1)
    adj_lst[v2-1].append(v1-1)
  for i in range(n):
    adj_lst[i] = list(set(adj_lst[i]))
  return (n, m, adj_lst)

def isConnectedByEdgeList(G):
  return isConnectedByAdjacencyList(edgeListToAdjacencyList(G))

def isConnectedByAdjacencyList(G):
  # 0 - index, adj_list repr
  n, m, V = G
  r = [False] * n
  q = Queue()
  q.put(0)
  r[0] = True
  while not q.empty():
    v = q.get()
    for adj in V[v]:
      if r[adj] == False:
        r[adj] = True
        q.put(adj)
  return all(r)

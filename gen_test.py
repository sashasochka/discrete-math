#!/usr/bin/python3
from random import *
n, m = map(int, input().split())
print(n, m)
for i in range(m):
  print(randint(1, n), randint(1, n))

#!/bin/bash
lab=lab1
filename="input.txt"
if [ $# -ge 1 ]; then
  filename=inputs/graph_$1.txt
fi
if [ $# -ge 2 ]; then
  lab=$2
fi
if [ ! -e $filename ]; then
  echo "File $filename doesn't exist!"
fi
if [ ! -x src/$lab.py ]; then
  echo "File src/$lab.py doesn't exist or is not executable"
fi 
echo -e "$filename\n\n" | src/$lab.py

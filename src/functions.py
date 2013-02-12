#!/usr/bin/env python3
import sys

def get_input_file():
  filename = input('Enter input filename or press enter to read from stdin: ')
  if filename.strip() == '':
    f = sys.stdin
  else:
    try:
      f = open(filename)
    except IOError:
      print('File doesn\'t exist')
      sys.exit(1)
  return f

def get_output_file():
  filename = input('Enter output filename or press enter to write to stdout: ')
  if filename.strip() == '':
    f = sys.stdout
  else:
    f = open(filename, 'w')
  return f

def print_matrix(matrix, fout, title = '', width = 3):
  n, m = len(matrix), len(matrix[0]) if len(matrix) else 0
  frmt = '{:>' + str(width) + '}'

  # first line
  fout.write(title + '|')
  for i in range(m):
    fout.write(frmt.format(i + 1))
  fout.write('\n')

  # second line
  fout.write('-' * (len(title) + width * (m+1)) + '\n')

  # subsequent lines
  for i in range(n):
    fout.write(('{:' + str(len(title)) + '}').format(i + 1) + '|')
    for j in range(m):
      fout.write(frmt.format(matrix[i][j]))
    fout.write('\n')

def print_vector(vector, fout, title = '', width = 3):
  frmt = '{:>' + str(width) + '}'
  n = len(vector)

  # first line
  fout.write(frmt.format('№') + '| ' + title + '\n')

  # second line
  fout.write('-' * (len(title) + width + 3) + '\n')

  # subsequent lines
  for i in range(n):
    fout.write(frmt.format(i + 1) + '|')
    fout.write(frmt.format(vector[i]))
    fout.write('\n')

def ask(title):
  answer = input('\n' + title + '\n')
  return answer.strip().lower() in ['yes', 'y', ''] 
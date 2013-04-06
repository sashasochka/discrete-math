"""
Common for all labs functions
"""
import sys


class IO:
    """
    IO class which does file initialization from
    either command line argumnts or stdin/stdout
    and provides a generilized way of printing to output file
    """
    def __init__(self):
        """
        Initialize input and ouput file objects from command line options or
        if absent from stdin/stdout
        """
        try:
            if len(sys.argv) > 1:
                self.filein = open(sys.argv[1], 'r')
            else:
                self.filein = sys.stdin

            if len(sys.argv) > 2:
                self.fileout = open(sys.argv[2], 'w')
            else:
                self.fileout = sys.stdout
        except IOError as e:
            print('Cannot open file {}!'.format(e.filename), file=sys.stderr)
            sys.exit(1)

    def print(self, *args: list, **kwargs: dict):
        """
        Works as builtin print except that prints to fileout
        """
        kwargs['file'] = self.fileout
        print(*args, **kwargs)

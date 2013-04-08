"""
Common for all labs functions
"""
import sys


class UserIO:
    """
    Class for talking to user through stdin/stdout
    """
    def readint(self, start: int=-sys.maxsize, stop: int=sys.maxsize,
                label: str='integer') -> int:
        """
        Read and return integer readed from input file after user prompt
        Return:
            read integer
        """
        val = None  # read integer
        repeat = True
        while repeat:
            print('Input {} (in range [{}; {}]: '.format(label, start,
                                                         stop - 1))
            try:
                val = int(input())
                repeat = False
            except ValueError:
                print('Cannot read as integer!')
        return val

    def readvertex(self, V: int, label: str='') -> int:
        """
        Read and return the number of vertex given there are V vertexes
        Return:
            0-based vertex index
        """
        return self.readint(1, V + 1, label + ' vertex number') - 1

    def print(self, *args, **kwargs):
        print(*args, **kwargs)


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
        self.sections = 0
        self.close_filein = False
        self.close_fileout = False
        try:
            if len(sys.argv) > 1:
                self.filein = open(sys.argv[1], 'r')
                self.close_filein = True
            else:
                self.filein = sys.stdin

            if len(sys.argv) > 2:
                self.fileout = open(sys.argv[2], 'w')
                self.close_fileout = True
            else:
                self.fileout = sys.stdout
        except IOError as e:
            print('Cannot open file {}!'.format(e.filename), file=sys.stderr)
            sys.exit(1)

    def readline(self) -> str:
        """
        Read and return string readed from input file
        Return:
            readed string
        """
        return self.filein.readline()

    def print(self, *args: list, **kwargs: dict):
        """
        Works as builtin print except that prints to fileout
        """
        kwargs['file'] = self.fileout
        print(*args, **kwargs)

    def section(self, title):
        self.sections += 1
        self.print('\n' + '=' * 40)
        self.print('{}) {}:\n'.format(self.sections, title))

    def print_numbered_list(self, lst: list, label: str='Value'):
        self.print('{:3} | {}'.format('№.', label))
        for i, val in enumerate(lst):
            self.print('{:3} | {}'.format(i + 1, val))
        self.print()

    def __del__(self):
        """
        Close opened files if needed
        """
        if self.close_filein:
            self.filein.close()
        if self.close_fileout:
            self.fileout.close()

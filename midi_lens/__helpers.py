import time
import sys


def print_line(message, tail='', width=50, fill='-', file=sys.stdout) -> None:
    """Print messate to certain width padded with fill character

    Args:
        message (string): string to be output
        tail (string): string at end of line (counts towards width, defaults to '')
        width (int, optional): Width of resulting string. Defaults to 50.
        fill (str, optional): Fill character. Defaults to '-'.
    """
    start = 2 * fill
    output = start + message + tail
    if len(tail):
        width -= 1
    padding = (width - (len(output) + 2)) * fill  # +2: two spaces
    print(start, message, padding, tail, file=file)


def log(outfile):
    def decorator(func):
        """Decorator: Log name of function being executed and
        the time it took for the function to run

        Args:
            func (function): function to be decorated
        """

        def wrapper(*args, **kwargs):

            print_line(func.__name__, "started", file=outfile)
            start_time = int(round(time.time() * 1000))
            result = func(*args, **kwargs)
            time_diff = int(round(time.time() * 1000)) - start_time

            time_string = '{} s'.format(round(time_diff / 1000, 3))
            print_line(func.__name__, time_string, file=outfile)
            print(result, file=outfile)
            print_line(func.__name__, "done", file=outfile)

            return result
        return wrapper
    return decorator


class Range:
    """Container to keep track of min and max values of numeric variables,
       updated with operand +=.
       Also keeps count of number of compared elements and
       their sum.
       At every possible step, the average can be retrieved
    """

    def __init__(self, min=float('inf'), max=0):
        self.min = min
        self.max = max
        self.size = 0
        self.sum = 0

    def __iadd__(self, val):
        """Update min, max, sum and count
        according to value

        Args:
            val (number): value to be compared
        Return:
            instance of self
        """
        self.size += 1
        self.sum += val
        self.min = min(self.min, val)
        self.max = max(self.max, val)
        return self

    def range(self):
        """Difference of curr min and max

        Returns:
            num: value of difference between min and max (always positive)
        """
        return self.max - self.min

    def avg(self):
        """Compute current average for all checked values
        If function is called when no values have been added,
        (i.e. size is 0) then return 0
        Returns:
            number: average of checked values
        """
        if self.size == 0:
            return 0
        return self.sum / self.size

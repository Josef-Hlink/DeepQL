""" Even smaller helper methods (split up to prevent circular imports). """

from datetime import datetime


def bold(text: str) -> str:
    """ Returns the passed text in bold. """
    return f'\033[1m{text}\033[0m'

def formatRuntime(seconds: float) -> str:
    """ Returns a formatted string of the passed runtime in (mm:)ss.fff. """
    if seconds < 60:
        return f'{seconds:.3f} sec'
    else:
        return datetime.utcfromtimestamp(seconds).strftime('%M:%S.%f')[:-3] + ' min'

class DotDict(dict):
    """ `dot.notation` access to dictionary attributes. """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

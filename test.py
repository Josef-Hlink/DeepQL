import psutil
import numpy as np


def getMemoryUsage():
    """ Returns the memory usage of the current process in MB. """
    return psutil.Process().memory_info().rss / (1024**2)

def test():
    before = getMemoryUsage()
    A = np.random.randint(0, 1000000, (1000, 5, 2), dtype=int)
    print(A.dtype)
    after = getMemoryUsage()
    print(f'Before: {before:.2f} MB')
    print(f'After: {after:.2f} MB')
    print(f'Difference: {after - before:.2f} MB')

if __name__ == '__main__':
    test()

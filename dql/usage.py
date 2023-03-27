#!/usr/bin/env python3

import argparse
from time import sleep

from .utils.parsewrapper import ParseWrapper
from .utils.helpers import fixDirectories, PrintIfVerbose, PrintIfDebug, prog

def main():
    fixDirectories()

    argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = ParseWrapper(argParser)()

    V, D = args['verbose'], args['debug']

    # Example usage of the PrintIfVerbose and PrintIfDebug classes
    # First we declare the classes, and then we can call them like the vanilla print function
    printV = PrintIfVerbose(V)
    printD = PrintIfDebug(D)
    # Normal print messages will not be affected
    print('This is a normal message')
    # Verbose messages will only be printed if the verbose flag is set through the command line
    printV('This is a verbose message')
    # Debug messages work similarly, but with a timestamp and a debug tag
    printD('This is a debug message')

    # Example usage of the prog function
    # Normally you would use a range object like this
    for _ in range(50):
        # Train for an epoch or something similar
        pass
    # Now we can pass a title to the prog function, and it will print a progress bar if verbose is set to true
    # Otherwise it will only print the title and run times (start timestamp & total time)
    for _ in prog(range(50), V, 'test'):
        sleep(0.1)
        pass


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import argparse
from time import sleep  # temporary

from .utils.parsewrapper import ParseWrapper
from .utils.helpers import fixDirectories
from .agents.DeepQLearningAgent import run

def main():
    
    fixDirectories()

    argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = ParseWrapper(argParser)()

    run(args)


if __name__ == '__main__':
    main()

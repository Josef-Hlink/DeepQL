#!/bin/bash

# e-greedy
dql -I e-greedy -ne 1000 -nr 5 -V -D

# boltzmann
dql -es bolzmann -I boltzmann -ne 1000 -nr 5 -V -D

# ucb
dql -es ucb -I ucb -ne 1000 -nr 5 -V -D

#!/bin/bash

# This script is used to run the experiments

# base
dql -I base -ne 1000 -nr 5 -V

# memory replay enabled
dql -MR -I replay -ne 1000 -nr 5 -V

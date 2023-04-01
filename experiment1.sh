#!/bin/bash

# This script is used to run the experiments

# base
dql -I base -ne 2000 -nr 10 -V

# memory replay enabled
dql -MR -I replay -ne 2000 -nr 10 -V

# target network enabled
dql -TN -I target -ne 2000 -nr 10 -V

# both target network and memory replay enabled
dql -TN -MR -I target-replay -ne 2000 -nr 10 -V

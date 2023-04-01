#!/bin/bash

# This script is used to run the experiments

# base
dql -I base -ne 2000 -nr 10

# memory replay enabled
dql -MR -I replay -ne 2000 -nr 10

# target network enabled
dql -TN -I target -ne 2000 -nr 10

# both target network and memory replay enabled
dql -TN -MR -I target-replay -ne 2000 -nr 10

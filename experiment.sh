#!/bin/bash

# base
dql -I base -ne 1000 -nr 5 -V -D
dql -I base2 -ne 1000 -nr 5 -V -D

# memory replay enabled
dql -MR -I replay -ne 1000 -nr 5 -V -D
dql -MR -I replay2 -ne 1000 -nr 5 -V -D

# target network enabled
dql -TN -I target -ne 1000 -nr 5 -V -D
dql -TN -I target2 -ne 1000 -nr 5 -V -D

# both target network and memory replay enabled
dql -TN -MR -I target-replay -ne 1000 -nr 5 -V -D
dql -TN -MR -I target-replay2 -ne 1000 -nr 5 -V -D

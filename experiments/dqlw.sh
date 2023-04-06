#!/bin/bash

# this script is just a wrapper for the dql command to facilitate multiple heavy runs (reps) of the same configuration
# this is done to avoid one python instance from building up a lot of garbage over multiple runs
# NOTE that you must pass the number of reps as the first argument (after -nr)!!!

# usage example:
# where you would normally do something like this:
# dql -ne 10000 -nr 10 -es egreedy -ER -I experiment_name
# you would do this:
# ./dqlw.sh -nr 10 -ne 10000 -es egreedy -ER -I experiment_name

REPS=$2
for i in $(seq 1 $REPS); do
    echo "Run $i"
    if [ $i -eq 0 ]; then
        # add verbose flag for first run to see config
        dql $@ -nr 1 -C -V
    fi
    dql $@ -nr 1 -C
done

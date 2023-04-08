#!/bin/bash

# exploring what happens when you make the agents take random actions for a while before starting to learn
# the idea is that this should make the agent avoid the local minima that it gets stuck in initially

DEFAULTS="-nr 5 -ne 20000 -bs 512 -V -g 0.999 -nw 10000"
NAME="RW"

echo "BASELINE"
./dqlw.sh $DEFAULTS -I ${NAME}-BL

echo "EXPERIENCE REPLAY"
./dqlw.sh $DEFAULTS -I ${NAME}-ER -ER -rb 100000

echo "TARGET NETWORK"
./dqlw.sh $DEFAULTS -I ${NAME}-TN -TN -tf 1000

echo "TARGET NETWORK + EXPERIENCE REPLAY"
./dqlw.sh $DEFAULTS -I ${NAME}-TR -TN -tf 1000 -ER -rb 100000

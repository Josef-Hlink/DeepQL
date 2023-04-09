#!/bin/bash

# a brute force approach; just throw compute power at it
# if the agent can reach an average reward of 40,
# this means it has done ±2M (40*50k) steps, which is quite a lot


DEFAULTS="-nr 6 -ne 50000 -bs 512 -V -g 0.999"
NAME="ABF"

echo "BASELINE"
./dqlw.sh $DEFAULTS -I ${NAME}-BL

echo "EXPERIENCE REPLAY"
./dqlw.sh $DEFAULTS -I ${NAME}-ER -ER -rb 100000

echo "TARGET NETWORK"
./dqlw.sh $DEFAULTS -I ${NAME}-TN -TN -tf 2500

echo "TARGET NETWORK + EXPERIENCE REPLAY"
./dqlw.sh $DEFAULTS -I ${NAME}-TR -TN -tf 2500 -ER -rb 100000

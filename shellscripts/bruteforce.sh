#!/bin/bash

# a brute force approach; just throw compute power at it
# if the agent can reach an average reward of 40,
# this means it has done ±2M (40*50k) steps, which is quite a lot

DEFAULTS="-nr 5 -ne 50000 -bs 512"
NAME="BF"

echo "BASELINE"
./dqlw.sh $DEFAULTS -I ${NAME}_BL

echo "EXPERIENCE REPLAY"
./dqlw.sh $DEFAULTS -I ${NAME}_ER -ER -rb 100000

echo "TARGET NETWORK"
./dqlw.sh $DEFAULTS -I ${NAME}_TN -TN -tf 2500

echo "TARGET NETWORK + EXPERIENCE REPLAY"
./dqlw.sh $DEFAULTS -I ${NAME}_TR -TN -tf 2500 -ER -rb 100000

#!/bin/bash

# seeing how the different exploration strategies hold up against each other
# we expect egreedy and boltzmann to be very similar, and ucb to be the worst
# from previous experiments we know that using a combination of ER and TN is the best,
# as well as using a random warmup, so we will use those here as well

DEFAULTS="-nr 5 -ne 20000 -bs 512 -V -g 0.999 -nw 10000 -TN -tf 1000 -ER -rb 100000"
NAME="ES"

echo "EGREEDY"
./dqlw.sh $DEFAULTS -I ${NAME}-EG -es egreedy

echo "BOLTZMANN"
./dqlw.sh $DEFAULTS -I ${NAME}-BM -es boltzmann

echo "UCB"
./dqlw.sh $DEFAULTS -I ${NAME}-UC -es ucb

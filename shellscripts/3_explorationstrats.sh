#!/bin/bash

# seeing how the different exploration strategies hold up against each other
# we expect egreedy and boltzmann to be very similar, and ucb to be the worst
# this is not because ucb is inherently bad, but because we have not had the time to tune it properly
# from previous experiments we know that using a combination of ER and TN is the best,
# as well as using a random warmup, so we will use those here as well


DEFAULTS="-nr 5 -ne 20000 -bs 512 -V -g 0.999 -nw 10000 -TN -tf 1000 -ER -rb 100000"
NAME="EA"

# we want to run both egreedy and boltzmann with annealing schemes 1 and 4
for i in 1 4; do
    echo "EGREEDY"
    ./dqlw.sh $DEFAULTS -I ${NAME}${i}-EG -es egreedy -as $i

    echo "BOLTZMANN"
    ./dqlw.sh $DEFAULTS -I ${NAME}${i}-BM -es boltzmann -as $i
done

echo "UCB"
# ucb uses a constant exploration factor, which is set in annealing scheme 0
./dqlw.sh $DEFAULTS -I ${NAME}0-UC -es ucb -as 0

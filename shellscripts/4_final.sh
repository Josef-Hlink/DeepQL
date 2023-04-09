#!/bin/bash

# as a grand finale, we perform one last experiment with the optimal settings
# that we found from the previous experiments:
# annealing scheme 4, both target network and experience replay, and a random warmup
# epsilon greedy turned out to be slightly more stable than boltzmann,
# at least for the experiments with 20k timesteps
# now we see how it holds up for 50k, 6 individual runs


./dqlw.sh -nr 6 -ne 50000 -bs 512 -V -g 0.999 -nw 10000 -TN -tf 1000 -ER -rb 100000 -I FIN-TR -es egreedy -as 4

#!/bin/bash

# if you just want to see if everything is still working, run this script
# it will run a total of 4 (2*2) repetitions with 100 episodes each
# all parameters are NOT default, so you can see if the results are still as expected

# -es boltzmann  # exploration strategy
# -as 2          # annealing scheme (id)
# -a 0.01        # alpha \ learning rate
# -g 0.9         # gamma \ discount factor
# -ne 100        # number of episodes
# -nr 2          # number of repetitions
# -nw 100        # number of random warmup steps
# -bs 16         # batch size
# -ER            # experience replay
# -rb 500        # replay buffer size
# -TN            # target network
# -tf 10         # target network update frequency
# -I test        # run id
# -C             # concat results
# -D             # debug
# -V             # verbose
# -R             # render

for i in {1..2}; do
    dql  -es boltzmann -as 2 -a 0.01 -g 0.9 -ne 100 -nr 2 -nw 100 -bs 16 -ER -rb 500 -TN -tf 10 -I test -C -D -V -R
done

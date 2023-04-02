#!/bin/bash

# with target

# target network enabled
dql -TN -I target -ne 1000 -nr 5 -V

# both target network and memory replay enabled
dql -TN -MR -I target-replay -ne 1000 -nr 5 -V

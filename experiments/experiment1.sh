#!/bin/bash

DEFAULTS="-nr 10 -ne 25000 -bs 512"

echo "base"
./dqlw.sh $DEFAULTS -I basef

echo "ER"
./dqlw.sh $DEFAULTS -I erf -ER -rb 100000

echo "TN"
./dqlw.sh $DEFAULTS -I tnf -TN -tf 2500

echo "TN+ER"
./dqlw.sh $DEFAULTS -I tnerf -TN -tf 2500 -ER -rb 100000

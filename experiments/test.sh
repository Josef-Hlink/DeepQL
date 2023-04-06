#!/bin/bash

for i in {1..5}; do
    dql -ne 2000 -nr 2 -V -I test -C
done

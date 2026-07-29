#!/usr/bin/env bash

cd $(dirname $0)

CONTROL=$1

shift

NODES=$*

me=$(getent hosts $(hostname -I) | awk '{print $2}' | grep -E '^node' | head -n1)

for n in $NODES
do
    echo $me
    ./wan-sim.py $CONTROL -H $n

    # assume nodeX access node0 with same interface as it does nodeY
    scp ./wan-sim.py $n:~/
    ssh $n python3 wan-sim.py $CONTROL -H $me
done

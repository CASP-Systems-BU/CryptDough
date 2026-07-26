#!/bin/bash

script_dir=$(dirname "$0")

cd "$script_dir/../../../../../build"

../scripts/run_experiment.py --build-only --no-division-correction -p 3 -r 192 -c mpi -s lan -t 0 -b -1 alexnet

mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &

# Pause for 300 milliseconds
sleep 0.5

mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &

# Pause for 300 milliseconds
sleep 0.5

mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca -n 3 --host node0,node1,node2 ./alexnet -b -1 -r 8 -s lan -x node -t 1 &

wait
#!/bin/bash

script_dir=$(dirname "$0")

cd "$script_dir/../../../../../build"

../scripts/run_experiment.py --build-only --no-division-correction -p 3 -r 192 -c mpi -s lan -t 0 -b -1 -neng 24 vgg16-imagenet-me

mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &

# Pause for 300 milliseconds
sleep 0.5

mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &

# Pause for 300 milliseconds
sleep 0.5

mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &
mpirun --mca btl_tcp_if_include 192.168.100.85/24 --mca oob_tcp_if_include 192.168.100.85/24 -n 3 --host node0,node1,node2 ./vgg16-imagenet -b -1 -r 8 -s lan -x node -t 1 &

wait
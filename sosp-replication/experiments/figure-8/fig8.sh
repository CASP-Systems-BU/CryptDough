#!/bin/bash

script_dir=$(dirname "$0")
run_script="$script_dir/../../../scripts/run_experiment.py"
log_dir="$script_dir/../../data/logs/fig8"

mkdir -p "$log_dir"

# Panels (a-b): GR (comparison), RCA, PPA via micro_primitives
$run_script -p 3 -r 268435456 -c nocopy -n 4 -s lan -t 5 -b -12 micro_primitives >> "$log_dir/lan-3pc-micro_primitives-32t.log"
$run_script -p 3 -r 268435456 -c nocopy -n 4 -s lan -t 4 -b -12 micro_primitives >> "$log_dir/lan-3pc-micro_primitives-16t.log"
$run_script -p 3 -r 268435456 -c nocopy -n 4 -s lan -t 3 -b -12 micro_primitives >> "$log_dir/lan-3pc-micro_primitives-8t.log"
$run_script -p 3 -r 268435456 -c nocopy -n 4 -s lan -t 2 -b -12 micro_primitives >> "$log_dir/lan-3pc-micro_primitives-4t.log"
$run_script -p 3 -r 268435456 -c nocopy -n 4 -s lan -t 1 -b -12 micro_primitives >> "$log_dir/lan-3pc-micro_primitives-2t.log"
$run_script -p 3 -r 268435456 -c nocopy -n 4 -s lan -t 0 -b -12 micro_primitives >> "$log_dir/lan-3pc-micro_primitives-1t.log"

# Panel (c): Conv2D, 256 instances.
$run_script -p 3 -r 256 -c nocopy -n 4 -s lan -t 5 -b -12 conv2d_model >> "$log_dir/lan-3pc-conv2d_model-32t.log"
$run_script -p 3 -r 256 -c nocopy -n 4 -s lan -t 4 -b -12 conv2d_model >> "$log_dir/lan-3pc-conv2d_model-16t.log"
$run_script -p 3 -r 256 -c nocopy -n 4 -s lan -t 3 -b -12 conv2d_model >> "$log_dir/lan-3pc-conv2d_model-8t.log"
$run_script -p 3 -r 256 -c nocopy -n 4 -s lan -t 2 -b -12 conv2d_model >> "$log_dir/lan-3pc-conv2d_model-4t.log"
$run_script -p 3 -r 256 -c nocopy -n 4 -s lan -t 1 -b -12 conv2d_model >> "$log_dir/lan-3pc-conv2d_model-2t.log"
$run_script -p 3 -r 256 -c nocopy -n 4 -s lan -t 0 -b -12 conv2d_model >> "$log_dir/lan-3pc-conv2d_model-1t.log"

# Panel (d): Sorting — bitonic and quicksort (64-bit), both 2^26.
$run_script -p 3 -r 67108864 -c nocopy -s lan -n 8 -t 5 -b -2 micro_sorting >> "$log_dir/lan-3pc-micro_sorting-32t.log"
$run_script -p 3 -r 67108864 -c nocopy -s lan -n 4 -t 4 -b -2 micro_sorting >> "$log_dir/lan-3pc-micro_sorting-16t.log"
$run_script -p 3 -r 67108864 -c nocopy -s lan -n 2 -t 3 -b -2 micro_sorting >> "$log_dir/lan-3pc-micro_sorting-8t.log"
$run_script -p 3 -r 67108864 -c nocopy -s lan -n 2 -t 2 -b -2 micro_sorting >> "$log_dir/lan-3pc-micro_sorting-4t.log"
$run_script -p 3 -r 67108864 -c nocopy -s lan -n 2 -t 1 -b -12 micro_sorting >> "$log_dir/lan-3pc-micro_sorting-2t.log"
$run_script -p 3 -r 67108864 -c nocopy -s lan -n 2 -t 0 -b -12 micro_sorting >> "$log_dir/lan-3pc-micro_sorting-1t.log"

#!/bin/bash

script_dir=$(dirname "$0")
run_script="$script_dir/../../../scripts/run_experiment.py"
log_dir="$script_dir/../../data/logs/table-3/cdough"

mkdir -p "$log_dir"

# LAN experiments with 2PC
$run_script -p 2 --no-division-correction -r 192 -c nocopy -n 1 -s lan -t 0 -b -12 alexnet >> "$log_dir/lan-2pc-alexnet.log"
$run_script -p 2 --no-division-correction -r 192 -c nocopy -n 1 -s lan -t 0 -b -12 vgg16 >> "$log_dir/lan-2pc-vgg16.log"
$run_script -p 2 --no-division-correction -r 8 -c nocopy -n 1 -s lan -t 0 -b -12 vgg16-imagenet >> "$log_dir/lan-2pc-vgg16-imagenet.log"

# WAN experiments with 2PC
$run_script -p 2 --no-division-correction -r 192 -c nocopy -n 1 -s wan -t 0 -b -1 alexnet >> "$log_dir/wan-2pc-alexnet.log"
$run_script -p 2 --no-division-correction -r 192 -c nocopy -n 1 -s wan -t 0 -b -1 vgg16 >> "$log_dir/wan-2pc-vgg16.log"
$run_script -p 2 --no-division-correction -r 8 -c nocopy -n 1 -s wan -t 0 -b -1 vgg16-imagenet >> "$log_dir/wan-2pc-vgg16-imagenet.log"
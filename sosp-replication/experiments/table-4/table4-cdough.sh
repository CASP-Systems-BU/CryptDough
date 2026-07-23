#!/bin/bash

script_dir=$(dirname "$0")
run_script="$script_dir/../../../scripts/run_experiment.py"
log_dir="$script_dir/../../data/logs/table-4/cdough"

mkdir -p "$log_dir"

# LAN experiments with MPSPDZ
$run_script -p 5 -npc 2 --no-division-correction -r 192 -c nocopy -n 16 -s lan -t 4 -b 27744 mpspdz_alexnet >> "$log_dir/lan-mpspdz-alexnet.log"

# WAN experiments with MPSPDZ
$run_script -p 5 -npc 2 --no-division-correction -r 192 -c nocopy -n 16 -s wan -t 4 -b -2 mpspdz_alexnet >> "$log_dir/wan-mpspdz-alexnet.log"
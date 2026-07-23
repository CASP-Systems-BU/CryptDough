#!/bin/bash

script_dir=$(dirname "$0")
run_script="$script_dir/../../../scripts/run_experiment.py"
log_dir="$script_dir/../../data/logs/fig7/fig7-cdough"

mkdir -p "$log_dir"

# LAN experiments with 3PC
$run_script -p 3 -r 4194304 -c nocopy -n 4 -s lan -t 4 -b -12  energy >> "$log_dir/lan-3pc-energy.log"
$run_script -p 3 -r 4194304 -c nocopy -n 4 -s lan -t 4 -b -12  medical >> "$log_dir/lan-3pc-medical.log"
$run_script -p 3 -r 4194304 -c nocopy -n 4 -s lan -t 4 -b -12  cloud >> "$log_dir/lan-3pc-cloud.log"

# LAN experiments with 4PC
$run_script -p 4 --sort bitonicsort -r 4194304 -c nocopy -n 4 -s lan -t 4 -b -12  energy >> "$log_dir/lan-4pc-energy.log"
$run_script -p 4 --sort bitonicsort -r 4194304 -c nocopy -n 4 -s lan -t 4 -b -12  medical >> "$log_dir/lan-4pc-medical.log"
$run_script -p 4 --sort bitonicsort -r 4194304 -c nocopy -n 4 -s lan -t 4 -b -12  cloud >> "$log_dir/lan-4pc-cloud.log"


# WAN experiments with 3PC
$run_script -p 3 -r 4194304 -c nocopy -n 16 -s wan -t 4 -b -1  energy >> "$log_dir/wan-3pc-energy.log"
$run_script -p 3 -r 4194304 -c nocopy -n 16 -s wan -t 4 -b -1  medical >> "$log_dir/wan-3pc-medical.log"
$run_script -p 3 -r 4194304 -c nocopy -n 16 -s wan -t 4 -b -1  cloud >> "$log_dir/wan-3pc-cloud.log"

# WAN experiments with 4PC
$run_script -p 4 --sort bitonicsort -r 4194304 -c nocopy -n 16 -s wan -t 4 -b -1  energy >> "$log_dir/wan-4pc-energy.log"
$run_script -p 4 --sort bitonicsort -r 4194304 -c nocopy -n 16 -s wan -t 4 -b -1  medical >> "$log_dir/wan-4pc-medical.log"
$run_script -p 4 --sort bitonicsort -r 4194304 -c nocopy -n 16 -s wan -t 4 -b -1  cloud >> "$log_dir/wan-4pc-cloud.log"
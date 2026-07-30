#!/bin/bash

script_dir=$(dirname "$0")
run_script="$script_dir/../../../scripts/run_experiment.py"
log_dir="$script_dir/../../data/logs/fig5"

mkdir -p "$log_dir"

# LAN Experiments
$run_script --no-division-correction --timeout 2000 -p 2 -r 16384 -c nocopy -n 4 -s lan -t 4 -b -2 patients_spo >> "$log_dir/lan-2pc.log"
$run_script --no-division-correction --timeout 2000 -p 3 -r 16384 -c nocopy -n 4 -s lan -t 4 -b -2 patients_spo >> "$log_dir/lan-3pc.log"
$run_script --no-division-correction --timeout 2000 --sort bitonicsort -p 4 -r 16384 -c nocopy -n 4 -s lan -t 4 -b -2 patients_spo >> "$log_dir/lan-4pc.log"
$run_script --no-division-correction --timeout 3000 --sort bitonicsort -p 5 -npc 2 -r 16384 -c nocopy -n 4 -s lan -t 4 -b -4 patients_spo >> "$log_dir/lan-spdz.log"


# WAN Experiments
$run_script --no-division-correction --timeout 2000 -p 2 -r 16384 -c nocopy -n 16 -s wan -t 4 -b -1 patients_spo >> "$log_dir/wan-2pc.log"
$run_script --no-division-correction --timeout 2000 -p 3 -r 16384 -c nocopy -n 16 -s wan -t 4 -b -1 patients_spo >> "$log_dir/wan-3pc.log"
$run_script --no-division-correction --timeout 3000 --sort bitonicsort -p 4 -r 16384 -c nocopy -n 16 -s wan -t 4 -b -1 patients_spo >> "$log_dir/wan-4pc.log"
$run_script --no-division-correction --timeout 10000 --sort bitonicsort -p 5 -npc 2 -r 16384 -c nocopy -n 16 -s wan -t 4 -b -4 patients_spo >> "$log_dir/wan-spdz.log"
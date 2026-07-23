#!/bin/bash

script_dir=$(dirname "$0")
log_dir="$script_dir/../../data/logs/fig6/orq"
mkdir -p "$log_dir"

baseline_dir="$script_dir/../../baselines/orq/build"
run_script="$baseline_dir/scripts/run_experiment.sh"

cd "$baseline_dir"

# LAN Experiments with 2PC
$run_script -p 2 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12  aspirin >> "$log_dir/lan-2pc-aspirin.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12  rcdiff >> "$log_dir/lan-2pc-rcdiff.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12  pwd-reuse >> "$log_dir/lan-2pc-pwd-reuse.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12  credit_score >> "$log_dir/lan-2pc-credit_score.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12  comorbidity >> "$log_dir/lan-2pc-comorbidity.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12  secrecy_q2 >> "$log_dir/lan-2pc-secrecy_q2.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12  market-share >> "$log_dir/lan-2pc-market-share.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12  custom_agg >> "$log_dir/lan-2pc-custom_agg.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12  distinct_patients >> "$log_dir/lan-2pc-distinct_patients.log"


# LAN Experiments with 3PC
$run_script -p 3 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12  aspirin >> "$log_dir/lan-3pc-aspirin.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12  rcdiff >> "$log_dir/lan-3pc-rcdiff.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12  pwd-reuse >> "$log_dir/lan-3pc-pwd-reuse.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12  credit_score >> "$log_dir/lan-3pc-credit_score.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12  comorbidity >> "$log_dir/lan-3pc-comorbidity.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12  secrecy_q2 >> "$log_dir/lan-3pc-secrecy_q2.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12  market-share >> "$log_dir/lan-3pc-market-share.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12  custom_agg >> "$log_dir/lan-3pc-custom_agg.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12  distinct_patients >> "$log_dir/lan-3pc-distinct_patients.log"

# WAN Experiments with 2PC
$run_script -p 2 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1  aspirin >> "$log_dir/wan-2pc-aspirin.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1  rcdiff >> "$log_dir/wan-2pc-rcdiff.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1  pwd-reuse >> "$log_dir/wan-2pc-pwd-reuse.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1  credit_score >> "$log_dir/wan-2pc-credit_score.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1  comorbidity >> "$log_dir/wan-2pc-comorbidity.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1  secrecy_q2 >> "$log_dir/wan-2pc-secrecy_q2.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1  market-share >> "$log_dir/wan-2pc-market-share.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1  custom_agg >> "$log_dir/wan-2pc-custom_agg.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1  distinct_patients >> "$log_dir/wan-2pc-distinct_patients.log"


# WAN Experiments with 3PC
$run_script -p 3 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1  aspirin >> "$log_dir/wan-3pc-aspirin.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1  rcdiff >> "$log_dir/wan-3pc-rcdiff.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1  pwd-reuse >> "$log_dir/wan-3pc-pwd-reuse.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1  credit_score >> "$log_dir/wan-3pc-credit_score.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1  comorbidity >> "$log_dir/wan-3pc-comorbidity.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1  secrecy_q2 >> "$log_dir/wan-3pc-secrecy_q2.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1  market-share >> "$log_dir/wan-3pc-market-share.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1  custom_agg >> "$log_dir/wan-3pc-custom_agg.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1  distinct_patients >> "$log_dir/wan-3pc-distinct_patients.log"
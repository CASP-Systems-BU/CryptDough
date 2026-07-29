#!/bin/bash

script_dir=$(dirname "$0")
build_dir="$script_dir/../../../build"


mkdir -p "$build_dir"
cd "$build_dir"

run_script="../scripts/run_experiment.py"
log_dir="../sosp-replication/data/logs/fig6/cdough"

mkdir -p "$log_dir"


# LAN Experiments with 2PC
$run_script -p 2 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12 --timeout 1000  aspirin >> "$log_dir/lan-2pc-aspirin.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12 --timeout 1000  rcdiff >> "$log_dir/lan-2pc-rcdiff.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12 --timeout 1000  pwd-reuse >> "$log_dir/lan-2pc-pwd-reuse.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12 --timeout 1000  credit_score >> "$log_dir/lan-2pc-credit_score.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12 --timeout 1000  comorbidity >> "$log_dir/lan-2pc-comorbidity.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12 --timeout 1000  secrecy_q2 >> "$log_dir/lan-2pc-secrecy_q2.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12 --timeout 1000  market-share >> "$log_dir/lan-2pc-market-share.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12 --timeout 1000  custom_agg >> "$log_dir/lan-2pc-custom_agg.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12 --timeout 1000  distinct_patients >> "$log_dir/lan-2pc-distinct_patients.log"


# LAN Experiments with 3PC
$run_script -p 3 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12 --timeout 1000  aspirin >> "$log_dir/lan-3pc-aspirin.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12 --timeout 1000  rcdiff >> "$log_dir/lan-3pc-rcdiff.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12 --timeout 1000  pwd-reuse >> "$log_dir/lan-3pc-pwd-reuse.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12 --timeout 1000  credit_score >> "$log_dir/lan-3pc-credit_score.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12 --timeout 1000  comorbidity >> "$log_dir/lan-3pc-comorbidity.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12 --timeout 1000  secrecy_q2 >> "$log_dir/lan-3pc-secrecy_q2.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12 --timeout 1000  market-share >> "$log_dir/lan-3pc-market-share.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12 --timeout 1000  custom_agg >> "$log_dir/lan-3pc-custom_agg.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s lan -t 4 -b -12 --timeout 1000  distinct_patients >> "$log_dir/lan-3pc-distinct_patients.log"

# WAN Experiments with 2PC
$run_script -p 2 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1 --timeout 2000  aspirin >> "$log_dir/wan-2pc-aspirin.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1 --timeout 2000  rcdiff >> "$log_dir/wan-2pc-rcdiff.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1 --timeout 2000  pwd-reuse >> "$log_dir/wan-2pc-pwd-reuse.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1 --timeout 2000  credit_score >> "$log_dir/wan-2pc-credit_score.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1 --timeout 2000  comorbidity >> "$log_dir/wan-2pc-comorbidity.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1 --timeout 2000  secrecy_q2 >> "$log_dir/wan-2pc-secrecy_q2.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1 --timeout 2000  market-share >> "$log_dir/wan-2pc-market-share.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1 --timeout 2000  custom_agg >> "$log_dir/wan-2pc-custom_agg.log"
$run_script -p 2 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1 --timeout 2000  distinct_patients >> "$log_dir/wan-2pc-distinct_patients.log"


# WAN Experiments with 3PC
$run_script -p 3 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1 --timeout 2000  aspirin >> "$log_dir/wan-3pc-aspirin.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1 --timeout 2000  rcdiff >> "$log_dir/wan-3pc-rcdiff.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1 --timeout 2000  pwd-reuse >> "$log_dir/wan-3pc-pwd-reuse.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1 --timeout 2000  credit_score >> "$log_dir/wan-3pc-credit_score.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1 --timeout 2000  comorbidity >> "$log_dir/wan-3pc-comorbidity.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1 --timeout 2000  secrecy_q2 >> "$log_dir/wan-3pc-secrecy_q2.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1 --timeout 2000  market-share >> "$log_dir/wan-3pc-market-share.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1 --timeout 2000  custom_agg >> "$log_dir/wan-3pc-custom_agg.log"
$run_script -p 3 -f 1 -c nocopy -n 4 -s wan -t 4 -b -1 --timeout 2000  distinct_patients >> "$log_dir/wan-3pc-distinct_patients.log"
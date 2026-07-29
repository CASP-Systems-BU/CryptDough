#!/bin/bash

script_dir=$(dirname "$0")
build_dir="$script_dir/../../build"

cd "$build_dir"

# Run all CDough experiments
../sosp-replication/experiments/run-all-cdough.sh

# Run all baseline experiments except for Piranha
../sosp-replication/experiments/run-all-baselines.sh

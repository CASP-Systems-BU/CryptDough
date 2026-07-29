#!/bin/bash

# This script runs all baseline experiments except for Piranha.

script_dir=$(dirname "$0")
build_dir="$script_dir/../../build"

cd "$build_dir"


# Running experiments for ORQ (Figure 6)
../sosp-replication/experiments/run-orq.sh

# Running experiments for TVA (Figure 7)
../sosp-replication/experiments/run-tva.sh

# Running experiments for Pigeon (Table 2)
../sosp-replication/experiments/run-pigeon.sh

# Running experiments for MPSPDZ (Table 4)
../sosp-replication/experiments/run-mpspdz.sh
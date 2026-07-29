#!/bin/bash

script_dir=$(dirname "$0")
build_dir="$script_dir/../../build"

cd "$build_dir"


# Running experiments for figure 5 to demonstrate multi-workload queries
../sosp-replication/experiments/figure-5/fig5.sh

# Running experiments for figure 6 to demonstrate comparison queries with ORQ
../sosp-replication/experiments/figure-6/fig6-cdough.sh

# Running experiments for figure 7 to demonstrate comparison queries with TVA
../sosp-replication/experiments/figure-7/fig7-cdough.sh

# Running experiments for figure 8 to demonstrate scalability benchmarking 
# for [Greater than - RCA - PPA - CONV2D - Bitonic Sort - QUICK Sort]
../sosp-replication/experiments/figure-8/fig8.sh

# Running experiments for table 2 to demonstrate comparison with Pigeon
../sosp-replication/experiments/table-2/table2-cdough.sh

# Running experiments for table 3 to demonstrate comparison with Piranha
../sosp-replication/experiments/table-3/table3-cdough.sh

# Running experiments for table 4 to demonstrate comparison with MPSPDZ
../sosp-replication/experiments/table-4/table4-cdough.sh
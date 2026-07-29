#!/bin/bash

script_dir=$(dirname "$0")
log_dir="$script_dir/../../data/logs/table-2/cdough"

# WAN simulation helper (same one run_experiment.py uses) and the peer nodes to
# shape. node0 is the local orchestrator, so only the remote parties are shaped;
# the 3PC experiments run across node0,node1,node2.
wan_sim="$script_dir/../../../scripts/profiling/comm/cluster-wan-sim.sh"
wan_nodes=(node1 node2)

mkdir -p "$log_dir"

# LAN experiments
$script_dir/ml/lan/ml_alexnet_3pc.sh >> "$log_dir/lan-3pc-alexnet.log"
$script_dir/ml/lan/ml_vgg16_3pc.sh >> "$log_dir/lan-3pc-vgg16.log"
$script_dir/ml/lan/ml_vgg16_imagenet_3pc.sh >> "$log_dir/lan-3pc-vgg16_imagenet.log"

# Turn on WAN simulation and make sure it is turned back off even if a WAN
# experiment fails or the script is interrupted.
"$wan_sim" on "${wan_nodes[@]}"
trap '"$wan_sim" off "${wan_nodes[@]}"' EXIT

# WAN experiments
$script_dir/ml/wan/ml_alexnet_3pc.sh >> "$log_dir/wan-3pc-alexnet.log"
$script_dir/ml/wan/ml_vgg16_3pc.sh >> "$log_dir/wan-3pc-vgg16.log"
$script_dir/ml/wan/ml_vgg16_imagenet_3pc.sh >> "$log_dir/wan-3pc-vgg16_imagenet.log"

# Turn off WAN simulation.
trap - EXIT
"$wan_sim" off "${wan_nodes[@]}"
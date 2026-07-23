#!/bin/bash

script_dir=$(dirname "$0")
log_dir="$script_dir/../../data/logs/fig7/tva"
mkdir -p "$log_dir"

baseline_dir="$script_dir/../../baselines/tva/build"
build_dir=$(cd "$baseline_dir" && pwd)
run_command="mpirun --host node0,node1,node2,node3 --mca btl_tcp_if_include 192.168.100.0/24"

# build_tva <np>: (Re)build the TVA energy/medical/cloud binaries for the given
# protocol on all participating parties. np=3 builds the 3PC semi-honest
# protocol on node0..node2; np=4 builds the 4PC malicious (Fantastic Four)
# protocol on node0..node3. Because 3PC and 4PC share one build directory, the
# directory is wiped and reconfigured on every call. Builds run in parallel over
# SSH and the function aborts if any node fails.
build_tva() {
    local np="$1"
    local cmake_args=""
    local hosts=(node0 node1 node2)
    if [ "$np" -eq 4 ]; then
        hosts=(node0 node1 node2 node3)
        cmake_args="-DPROTOCOL_VAR=-DUSE_FANTASTIC_FOUR"
    fi

    echo "==> Building TVA ${np}PC on: ${hosts[*]}"
    local pids=()
    for host in "${hosts[@]}"; do
        ssh "$host" "cd '${build_dir}' && rm -rf * && cmake .. ${cmake_args} && make -j energy medical cloud" &
        pids+=("$!")
    done

    local failed=0
    for i in "${!hosts[@]}"; do
        if ! wait "${pids[$i]}"; then
            echo "!! TVA ${np}PC build FAILED on ${hosts[$i]}" >&2
            failed=1
        fi
    done

    if [ "$failed" -ne 0 ]; then
        echo "==> Aborting: one or more TVA ${np}PC builds failed." >&2
        exit 1
    fi
    echo "==> TVA ${np}PC build complete on all nodes."
}

cd "$build_dir"


build_tva 3
# LAN experiments with 3PC
$run_command -np 3 energy 16 1 4096 4194304 >> "$log_dir/lan-3pc-energy.log"
$run_command -np 3 medical 16 1 4096 4194304 >> "$log_dir/lan-3pc-medical.log"
$run_command -np 3 cloud 16 1 4096 4194304 >> "$log_dir/lan-3pc-cloud.log"


build_tva 4
# LAN experiments with 4PC
$run_command -np 4 energy 16 1 4096 4194304 >> "$log_dir/lan-4pc-energy.log"
$run_command -np 4 medical 16 1 4096 4194304 >> "$log_dir/lan-4pc-medical.log"
$run_command -np 4 cloud 16 1 4096 4194304 >> "$log_dir/lan-4pc-cloud.log"

# TODO: turn on WAN

build_tva 3
# WAN experiments with 3PC
$run_command -np 3 energy 16 1 65536 4194304 >> "$log_dir/wan-3pc-energy.log"
$run_command -np 3 medical 16 1 65536 4194304 >> "$log_dir/wan-3pc-medical.log"
$run_command -np 3 cloud 16 1 65536 4194304 >> "$log_dir/wan-3pc-cloud.log"

build_tva 4
# WAN experiments with 4PC
$run_command -np 4 energy 16 1 65536 4194304 >> "$log_dir/wan-4pc-energy.log"
$run_command -np 4 medical 16 1 65536 4194304 >> "$log_dir/wan-4pc-medical.log"
$run_command -np 4 cloud 16 1 65536 4194304 >> "$log_dir/wan-4pc-cloud.log"
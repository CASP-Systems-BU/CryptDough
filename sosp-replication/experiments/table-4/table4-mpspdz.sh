#!/bin/bash
#
# table4-mpspdz.sh — Run the MP-SPDZ (SPDZ2k) AlexNet inference baseline for
# Table 4, orchestrated from node0.
#
# Table 4 compares CryptDough against MP-SPDZ. This script runs ONLY the MP-SPDZ
# side (the CryptDough side is table4-cdough.sh). It compiles the AlexNet CIFAR-10
# inference program (192 images, 16 threads) and runs it under the SPDZ2k 2PC
# protocol, with:
#   * party 0 -> node0
#   * party 1 -> node1
#   * host (-h) -> node0
# The two parties' output is written per party into the Table 4 MP-SPDZ log
# directory (data/logs/table-4/mpspdz).
#
# Prerequisites (already handled by the artifact setup):
#   * MP-SPDZ is installed at sosp-replication/baselines/mpspdz on every node
#     (see sosp-replication/setup/setup_mpspdz.sh), including the SPDZ2k source
#     edits documented there.
#   * node0 has SSH access to node0 and node1, and the repository is deployed to
#     the same absolute path on both nodes.
#
# Tunables (env vars):
#   USE_F  1 (default) passes -F and generates fake offline material with
#          Fake-Offline.x; 0 runs without -F. Whether -F reproduces the paper's
#          28.45 s LAN / 581.9 s WAN is an empirical open item, so it is kept
#          overridable here rather than hard-coded.
#   RUNS   Number of repetitions per setting (default 1).
#
# Note: 3PC segfaults at 192 images, so this comparison is 2PC only.
#
# Usage (from node0):
#   ./sosp-replication/experiments/table-4/table4-mpspdz.sh
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# MP-SPDZ install directory and the Table 4 log directory. The repository is
# deployed to the same absolute path on every node, so mpspdz_dir is valid on
# each node reached over SSH. sosp-replication (script_dir/../..) always exists,
# so we can resolve the log dir's parent before the log dir itself is created.
mpspdz_dir="$(cd "$script_dir/../../baselines/mpspdz" && pwd)"
log_dir="$(cd "$script_dir/../.." && pwd)/data/logs/table-4/mpspdz"
mkdir -p "$log_dir"

# WAN emulation script shared with the CryptDough side (run_experiment.py -s wan).
# The repository is deployed to the same absolute path on both nodes, so the
# same path is valid on node1 over SSH. cluster-wan-sim.sh is not used here
# because it identifies node0 by `hostname --short`, which the other node cannot
# resolve on this cluster; the node0/node1 aliases are used instead (the same
# reasoning as table3-piranha.sh).
wan_sim_py="$(cd "$script_dir/../../../scripts/profiling/comm" && pwd)/wan-sim.py"

USE_F=${USE_F:-1}
RUNS=${RUNS:-1}
PROGRAM="torch_cifar_alex_infer_single-192-16"

# 2PC party-to-node mapping and the host address both parties dial.
HOST=node0
NODES=(node0 node1) # NODES[0]=party 0, NODES[1]=party 1

FFLAG=""
[ "$USE_F" = "1" ] && FFLAG="-F"

# build_mpspdz: compile the AlexNet inference program (and, with -F, generate the
# fake offline material) on BOTH parties. MP-SPDZ needs the compiled schedule and
# bytecode present on every machine that runs a party, so — mirroring
# fig7-tva.sh's build-on-all-parties approach — the compile is fanned out over
# SSH to node0 and node1 in parallel. Aborts if any node fails.
#
# compile.py is run with the MP-SPDZ .env virtualenv interpreter (created by
# setup_mpspdz.sh) because the AlexNet program imports numpy/torch/torchvision.
build_mpspdz() {
    local build_cmd="cd '${mpspdz_dir}' && ./.env/bin/python ./compile.py -R 64 --budget=100000 torch_cifar_alex_infer_single 192 16"
    if [ "$USE_F" = "1" ]; then
        build_cmd="${build_cmd} && ./Fake-Offline.x 2 -Z 64 -S 64 -p ${PROGRAM}"
    fi

    echo "==> Building MP-SPDZ program on: ${NODES[*]} (USE_F=${USE_F})"
    local pids=()
    for host in "${NODES[@]}"; do
        ssh "$host" "$build_cmd" &
        pids+=("$!")
    done

    local failed=0
    for i in "${!NODES[@]}"; do
        if ! wait "${pids[$i]}"; then
            echo "!! MP-SPDZ build FAILED on ${NODES[$i]}" >&2
            failed=1
        fi
    done

    if [ "$failed" -ne 0 ]; then
        echo "==> Aborting: one or more MP-SPDZ builds failed." >&2
        exit 1
    fi
    echo "==> MP-SPDZ build complete on all nodes."
}

# run_mpspdz <env-label> <log-prefix>: run the SPDZ2k 2PC inference RUNS times,
# launching party 0 (node0) in the background and party 1 (node1) in the
# foreground so both parties are up at the same time to connect. Each party's
# output is appended to its own per-run log under log_dir.
run_mpspdz() {
    local env_label="$1" log_prefix="$2"
    for run in $(seq 1 "$RUNS"); do
        echo "=== MP-SPDZ AlexNet | 2PC ${env_label} | run ${run}/${RUNS} ==="
        ssh "${NODES[0]}" "cd '${mpspdz_dir}' && ./spdz2k-party.x -N 2 -p 0 ${FFLAG} -h ${HOST} ${PROGRAM}" \
            >> "${log_dir}/${log_prefix}-party0-run${run}.log" 2>&1 &
        local p0=$!
        sleep 2
        ssh "${NODES[1]}" "cd '${mpspdz_dir}' && ./spdz2k-party.x -N 2 -p 1 ${FFLAG} -h ${HOST} ${PROGRAM}" \
            >> "${log_dir}/${log_prefix}-party1-run${run}.log" 2>&1
        wait "$p0"
        sleep 5
    done
}

# Apply the WAN emulation state on both directions of the node0<->node1 link:
# locally toward node1 and on node1 toward node0. Each side auto-detects its
# interface from the (resolvable) peer alias. Needs passwordless sudo for tc on
# both nodes (tc netem: 6 Gbit rate + 10 ms per-link delay = 20 ms RTT).
set_wan() { # $1 = on|off
    local state="$1"
    python3 "$wan_sim_py" "$state" -H "${NODES[1]}" &&
        ssh "${NODES[1]}" "python3 '${wan_sim_py}' ${state} -H ${NODES[0]}"
}

# Always disable WAN emulation on exit so an aborted WAN half can never corrupt
# later measurements.
wan_enabled=0
disable_wan() {
    if [ "$wan_enabled" -eq 1 ]; then
        echo "==> Disabling WAN emulation (cleanup)..."
        set_wan off
        wan_enabled=0
    fi
}
trap disable_wan EXIT

build_mpspdz

# LAN experiments with 2PC
run_mpspdz "LAN" "lan-2pc-alexnet"

# Enable WAN emulation between node0 and node1 for the WAN half.
echo "==> Enabling WAN emulation (6 Gbit, 20 ms RTT) between ${NODES[0]} and ${NODES[1]}..."
wan_enabled=1
if ! set_wan on; then
    echo "!! Failed to enable WAN emulation; aborting before the WAN half." >&2
    exit 1
fi

# WAN experiments with 2PC
run_mpspdz "WAN" "wan-2pc-alexnet"

# Turn off WAN emulation.
disable_wan

echo ""
echo "Results in $log_dir"
echo "MP-SPDZ timing: grep 'Time =' ${log_dir}/*-party0-*.log"

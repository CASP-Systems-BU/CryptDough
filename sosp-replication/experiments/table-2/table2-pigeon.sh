#!/bin/bash
#
# table2-pigeon.sh — Run the Pigeon (chart21/hpmpc) ML-inference experiments for
# Table 2, orchestrated from node0.
#
# Table 2 compares CryptDough against Pigeon under model parallelism. This script
# runs the two Pigeon experiments end-to-end from node0, fanning out over SSH to
# the participating nodes (mirroring sosp-replication/setup/setup_pigeon.sh):
#   * 3PC (SPLITROLES=1, PROTOCOL=5):  parties 0,1,2   -> node0,node1,node2
#   * 4PC (SPLITROLES=3, PROTOCOL=12): parties 0,1,2,3 -> node0,node1,node2,node3
# Pigeon's scripts/run.sh dispatches SPLITROLES=1 to split-roles-3-execute.sh
# (3 parties) and SPLITROLES=3 to split-roles-4-execute.sh (4 parties), so the
# 3PC experiment uses 3 nodes and the 4PC experiment uses 4 nodes.
#
# This script resolves the peer IPs from the node aliases, derives each node's
# PID automatically, and runs make + scripts/run.sh from the Pigeon install
# directory. Each node writes its own per-party log into the log directory.
#
# Prerequisites (already handled by the artifact setup):
#   * Pigeon is installed at sosp-replication/baselines/hpmpc on every node
#     (see sosp-replication/setup/setup_pigeon.sh).
#   * node0 has SSH access to node0..node3 and /etc/hosts maps node0..node3 to
#     their IPs (see scripts/setup/_update_hostfile.sh).
#
# Usage (from node0):
#   ./sosp-replication/experiments/table-2/table2-pigeon.sh
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pigeon install directory and the Table 2 log directory. The repository is
# deployed to the same absolute path on every node, so these paths are valid on
# each node reached over SSH. sosp-replication (script_dir/../..) always exists,
# so we can resolve it before the log dir itself is created.
hpmpc_dir="$(cd "$script_dir/../../baselines/hpmpc" && pwd)"
log_dir="$(cd "$script_dir/../.." && pwd)/data/logs/table2/pigeon"

# Node aliases; party N runs on node N.
NODES=(node0 node1 node2 node3)

# Resolve a node alias/host to a dotted-decimal IPv4 address. hpmpc's Connect()
# (core/networking/socket.hpp) resolves peers with inet_aton(), which accepts
# ONLY dotted-decimal IPv4 and rejects hostnames; passing an alias makes the
# connecting parties abort with "Invalid address". So every peer address handed
# to scripts/run.sh must be a resolved IP.
resolve_ip() {
    local host="${1#*@}" # strip any "user@" prefix
    if [[ "$host" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        printf '%s' "$host" # already a dotted-decimal IPv4 address
        return 0
    fi
    local ip
    ip="$(python3 -c 'import socket, sys; print(socket.gethostbyname(sys.argv[1]))' "$host" 2>/dev/null)" || true
    if [[ -z "$ip" ]]; then
        echo "!! Could not resolve host '$host' to an IPv4 address." >&2
        echo "   hpmpc requires dotted-decimal IPs for the party-to-party connection." >&2
        exit 1
    fi
    printf '%s' "$ip"
}

IP0="$(resolve_ip "${NODES[0]}")"; IP1="$(resolve_ip "${NODES[1]}")"
IP2="$(resolve_ip "${NODES[2]}")"; IP3="$(resolve_ip "${NODES[3]}")"
echo "==> Wire IPs: P0=${IP0} P1=${IP1} P2=${IP2} P3=${IP3}"

mkdir -p "$log_dir"

# Run one Pigeon experiment across its participating nodes.
#   $1 label        - short name used for console output
#   $2 num_parties  - number of participating parties/nodes (3 or 4)
#   $3 make_flags   - the experiment-specific make configuration (everything
#                     except PARTY, which is set per node)
#   $4 splitroles   - SPLITROLES identifier passed to scripts/run.sh (-s)
#   $5 log_name     - log file name for this experiment (relative to log_dir),
#                     e.g. "lan-3pc.log"; each participating node writes its own
#                     copy at this same path
#
# The build and run are split into two phases so that every party finishes
# building before any party starts running; otherwise a party that builds first
# would launch scripts/run.sh and fail to connect to peers still compiling.
run_experiment() {
    local label="$1" num_parties="$2" make_flags="$3" splitroles="$4" log_name="$5"
    local last_party=$((num_parties - 1))
    local logf="${log_dir}/${log_name}"

    echo
    echo "======================================================================"
    echo "==> ${label}: building on parties 0..${last_party} (${NODES[*]:0:$num_parties})"

    # Phase 1: build the party executables on every participating node in
    # parallel; each node's log is truncated here so the run output appends to a
    # fresh file.
    local build_pids=() i
    for ((i = 0; i < num_parties; i++)); do
        ssh "${NODES[$i]}" "bash -s" >"$logf" 2>&1 <<REMOTE &
set -uo pipefail
cd "${hpmpc_dir}"
echo "[party ${i}] Building: make -j ${make_flags} PARTY=${i}"
make -j ${make_flags} PARTY=${i}
REMOTE
        build_pids+=("$!")
    done

    local build_failed=0
    for ((i = 0; i < num_parties; i++)); do
        if ! wait "${build_pids[$i]}"; then
            echo "!! ${label}: build FAILED on party ${i} (${NODES[$i]}). See ${logf} on that host." >&2
            build_failed=1
        fi
    done
    if [[ "$build_failed" -ne 0 ]]; then
        echo "!! ${label}: aborting before the run phase because a build failed." >&2
        return 1
    fi
    echo "==> ${label}: all ${num_parties} parties built."

    # Phase 2: launch scripts/run.sh on every participating node simultaneously,
    # appending to the same per-party log. Stale party processes from a prior run
    # are cleared first so their ports are free.
    echo "==> ${label}: running on parties 0..${last_party} simultaneously..."
    local run_pids=()
    for ((i = 0; i < num_parties; i++)); do
        ssh "${NODES[$i]}" "bash -s" >>"$logf" 2>&1 <<REMOTE &
set -uo pipefail
cd "${hpmpc_dir}"
pkill -9 -f run-P 2>/dev/null || true
echo "[party ${i}] Running: scripts/run.sh -p ${i} -s ${splitroles} -n 4 -a ${IP0} -b ${IP1} -c ${IP2} -d ${IP3}"
scripts/run.sh -p ${i} -s ${splitroles} -n 4 -a ${IP0} -b ${IP1} -c ${IP2} -d ${IP3}
REMOTE
        run_pids+=("$!")
    done

    local run_failed=0
    for ((i = 0; i < num_parties; i++)); do
        if ! wait "${run_pids[$i]}"; then
            echo "!! ${label}: run FAILED on party ${i} (${NODES[$i]}). See ${logf} on that host." >&2
            run_failed=1
        fi
    done
    if [[ "$run_failed" -ne 0 ]]; then
        return 1
    fi
    echo "==> ${label}: done. Log: ${logf} on each node."
}

overall=0

# LAN Experiments with 3PC
run_experiment "3pc" 3 \
    "NUM_INPUTS=1 PROCESS_NUM=4 SPLITROLES=1 DATTYPE=512 BITLENGTH=64 FUNCTION_IDENTIFIER=180 PROTOCOL=5" \
    1 "lan-3pc-alexnet.log" || overall=1

run_experiment "3pc" 3 \
    "NUM_INPUTS=1 PROCESS_NUM=4 SPLITROLES=1 DATTYPE=512 BITLENGTH=64 FUNCTION_IDENTIFIER=174 PROTOCOL=5" \
    1 "lan-3pc-vgg16.log" || overall=1

run_experiment "3pc" 3 \
    "NUM_INPUTS=1 PROCESS_NUM=4 SPLITROLES=1 DATTYPE=512 BITLENGTH=64 FUNCTION_IDENTIFIER=179 PROTOCOL=5" \
    1 "lan-3pc-vgg16_imagenet.log" || overall=1


# TODO: Turn on WAN 
# WAN Experiments with 3PC

run_experiment "3pc" 3 \
    "NUM_INPUTS=1 PROCESS_NUM=4 SPLITROLES=1 DATTYPE=512 BITLENGTH=64 FUNCTION_IDENTIFIER=180 PROTOCOL=5" \
    1 "wan-3pc-alexnet.log" || overall=1

run_experiment "3pc" 3 \
    "NUM_INPUTS=1 PROCESS_NUM=4 SPLITROLES=1 DATTYPE=512 BITLENGTH=64 FUNCTION_IDENTIFIER=174 PROTOCOL=5" \
    1 "wan-3pc-vgg16.log" || overall=1

run_experiment "3pc" 3 \
    "NUM_INPUTS=1 PROCESS_NUM=4 SPLITROLES=1 DATTYPE=512 BITLENGTH=64 FUNCTION_IDENTIFIER=179 PROTOCOL=5" \
    1 "wan-3pc-vgg16_imagenet.log" || overall=1

echo
echo "======================================================================"
if [[ "$overall" -ne 0 ]]; then
    echo "One or more Pigeon experiments failed. Inspect the per-party logs under:"
    echo "  ${log_dir}"
    exit 1
fi
echo "SUCCESS: all Pigeon experiments completed."
echo "======================================================================"
#!/usr/bin/env bash
#
# setup_tva.sh — Install TVA (CASP-Systems-BU/tva) on an MPC cluster.
#
# For each supplied host this script, over SSH and in parallel:
#   1. Clones the TVA repository at the pinned commit into the SAME absolute
#      path on every node (sosp-replication/baselines/tva).
#   2. Creates the build/ directory inside the repo and runs `cmake ..`,
#      selecting the protocol from the node count: 3 nodes build the 3PC
#      semi-honest protocol (plain `cmake ..`), 4 nodes build the 4PC
#      malicious Fantastic Four protocol (`cmake .. -DPROTOCOL_VAR=...`).
#   3. Builds the `main` test binary (`make main`).
#
# Once every node has built, it runs TVA's simple example across the cluster
# from node0:
#
#   mpirun --host <node0>,<node1>,... -np <#nodes> ./main
#
# (TVA's README runs the single-node simple example as `mpirun -np 3 ./main`;
# here we generalize it to the cluster by launching one process per node via
# `--host`, which requires node0 to have SSH access to the other nodes.)
#
# This script does NOT install system dependencies. Each node is expected to
# already provide TVA's requirements: a C++14 compiler, CMake (>=3.15.0),
# pkg-config (>=0.29.2), Libsodium (>=1.0.18), and an MPI implementation
# (OpenMPI or MPICH).
#
# Usage:
#   ./setup_tva.sh <node0> [node1 ...]
#
# TVA's `main` example is a 3- or 4-party protocol, so exactly 3 nodes (3PC) or
# 4 nodes (4PC) must be given. Each <nodeN> is an IP address, "user@IP", or an
# SSH alias (e.g. from ~/.ssh/config) reachable via `ssh <nodeN>`. node0 (the
# first argument) is the main node and must be able to SSH into the others so
# that mpirun can launch the cluster example.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <node0> [node1 ...]" >&2
    echo "  node0 (first argument) is the main node and must have SSH access to the others." >&2
    echo "  Provide 3 nodes for the 3PC protocol or 4 nodes for the 4PC protocol." >&2
    exit 1
fi

NODES=("$@")

REPO_URL="https://github.com/CASP-Systems-BU/tva"
REPO_COMMIT="e777cbadd28ce5da7d98c657aca64275bac164ac"

# Resolve the install location from this script's own location so it never
# depends on the caller's working directory. The script lives in
# sosp-replication/setup/, so the baselines dir is one level up. TVA is
# installed into baselines/tva, and this absolute path is used on every node.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "${SCRIPT_DIR}/../baselines" && pwd)"
INSTALL_DIR="${BASELINES_DIR}/tva"

# Bare host list for `mpirun --host` (strip any "user@" SSH prefix), plus the
# process count (one MPI process per supplied node).
HOST_LIST=""
for node in "${NODES[@]}"; do
    HOST_LIST+="${node#*@},"
done
HOST_LIST="${HOST_LIST%,}"
NPROC="${#NODES[@]}"

# Select the TVA protocol from the node count. TVA chooses the protocol at
# cmake-configure time via PROTOCOL_VAR: the default build is the 3PC
# semi-honest protocol, and defining USE_FANTASTIC_FOUR builds the 4PC
# malicious Fantastic Four protocol.
case "${NPROC}" in
    3)
        PROTOCOL_NAME="3PC (semi-honest)"
        CMAKE_PROTOCOL_ARGS=""
        ;;
    4)
        PROTOCOL_NAME="4PC (malicious, Fantastic Four)"
        CMAKE_PROTOCOL_ARGS="-DPROTOCOL_VAR=-DUSE_FANTASTIC_FOUR"
        ;;
    *)
        echo "!! TVA's 'main' example supports only 3 nodes (3PC) or 4 nodes (4PC); got ${NPROC}." >&2
        exit 1
        ;;
esac
echo "==> ${NPROC} node(s) supplied -> building the ${PROTOCOL_NAME} protocol."

LOG_FILE="~/install_tva.log"
echo "==> Each node writes its output to ${LOG_FILE} on its own host."

# Clone + configure + build the example on every node, in parallel.

remote_build_script() {
    cat <<REMOTE
set -euo pipefail

echo "[tva] Cloning ${REPO_URL} into ${INSTALL_DIR}..."
mkdir -p "$(dirname "${INSTALL_DIR}")"
if [[ ! -d "${INSTALL_DIR}/.git" ]]; then
    git clone "${REPO_URL}" "${INSTALL_DIR}"
fi
cd "${INSTALL_DIR}"
git fetch --all --tags
git checkout ${REPO_COMMIT}
git submodule update --init --recursive

echo "[tva] Creating build dir and running 'cmake ..' for the ${PROTOCOL_NAME} protocol..."
mkdir -p build
cd build
cmake .. ${CMAKE_PROTOCOL_ARGS}

echo "[tva] Building the 'main' example ('make main')..."
make main

echo "[tva] Build complete."
REMOTE
}

echo "==> Phase 1/2: cloning + configuring + building on ${NPROC} node(s) in parallel..."
build_pids=()
for i in "${!NODES[@]}"; do
    echo "    - node $i -> ${NODES[$i]}"
    ssh "${NODES[$i]}" "bash -s > ${LOG_FILE} 2>&1" \
        <<<"$(remote_build_script)" &
    build_pids+=("$!")
done

build_failed=0
for i in "${!NODES[@]}"; do
    if ! wait "${build_pids[$i]}"; then
        echo "!! Build FAILED on node ${i} (${NODES[$i]}). See ${LOG_FILE} on that host." >&2
        build_failed=1
    else
        echo "    node ${i} build OK"
    fi
done

if [[ "$build_failed" -ne 0 ]]; then
    echo "==> One or more builds failed; aborting before the cluster example." >&2
    exit 1
fi
echo "==> All ${NPROC} node(s) built successfully."

# Run TVA's simple example across the cluster from node0.

remote_run_script() {
    cat <<REMOTE
set -uo pipefail
cd "${INSTALL_DIR}/build"
echo "[tva] Running: mpirun --host ${HOST_LIST} -np ${NPROC} ./main"
mpirun --host ${HOST_LIST} -np ${NPROC} ./main
REMOTE
}

echo "==> Phase 2/2: launching the cluster example from node0 (${NODES[0]})..."
if ssh "${NODES[0]}" "bash -s" <<<"$(remote_run_script)"; then
    run_ok=1
else
    run_ok=0
fi

echo
echo "======================================================================"
if [[ "$run_ok" -ne 1 ]]; then
    echo "TVA installed on all nodes, but the cluster example did not complete."
    echo "Inspect the output above and ${LOG_FILE} on each host."
    exit 1
fi
echo "SUCCESS: TVA cloned into ${INSTALL_DIR} on all nodes, built, and the"
echo "cluster example ran across: ${NODES[*]}"
echo "======================================================================"

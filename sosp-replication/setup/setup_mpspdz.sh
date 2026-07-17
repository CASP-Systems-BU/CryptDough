#!/usr/bin/env bash
#
# setup_mpspdz.sh — Install MP-SPDZ (data61/MP-SPDZ) on an MPC cluster.
#
# For each supplied host this script, over SSH and in parallel:
#   1. Installs ONLY the MP-SPDZ build dependencies that CryptDough's
#      scripts/setup/_setup_required.sh does not already provide (see
#      MISSING_DEPS below for the derivation), plus the two extras needed by
#      MP-SPDZ's remote runner.
#   2. Clones the MP-SPDZ repository at the pinned commit into the SAME absolute
#      path on every node (sosp-replication/baselines/mpspdz).
#   3. Runs MP-SPDZ's `make setup` — the install step from the README's
#      "TL;DR (Source from GitHub)" that builds the bundled crypto dependencies
#      (libOTe, SimpleOT, and Boost locally where needed).
#   4. Builds the MASCOT virtual machine (`make -j mascot-party.x`), the VM used
#      by the README's simple tutorial example.
#
# Once every node has built, it runs MP-SPDZ's simple tutorial example ACROSS
# the cluster from node0 using MP-SPDZ's own remote runner:
#
#   Scripts/compile-run.py -H HOSTS -E mascot tutorial
#
# HOSTS lists every node and points each one at the absolute install path.
# compile-run.py compiles the tutorial and MASCOT VM once on node0, then uploads
# the binary, inputs, and freshly generated SSL certificates to every node over
# SSH and launches the parties (this is the README's "one-command remote
# execution"). It therefore requires node0 to have SSH access to the other
# nodes and the Fabric Python library (python3-fabric, installed below).
#
# This mirrors MP-SPDZ's README "simple example" (the `tutorial` compiled and
# run with MASCOT), generalized from a single machine to the cluster via
# `-H HOSTS`.
#
# Usage:
#   ./setup_mpspdz.sh <node0> [node1 ...]
#
# Any number of nodes (>= 2) may be given; the party count equals the node
# count. Each <nodeN> is an IP address, "user@IP", or an SSH alias (e.g. from
# ~/.ssh/config) reachable via `ssh <nodeN>`. node0 (the first argument) is the
# main node and must be able to SSH into the others so the tutorial can run
# across the cluster.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <node0> [node1 ...]" >&2
    echo "  Provide at least 2 nodes; the party count equals the node count." >&2
    echo "  node0 (first argument) is the main node and must have SSH access to the others." >&2
    exit 1
fi

NODES=("$@")

REPO_URL="https://github.com/data61/MP-SPDZ"
REPO_COMMIT="ae3fb09d905f1d7280dc3e4b4aca3154b0cab89b"

# Resolve the install location from this script's own location so it never
# depends on the caller's working directory. The script lives in
# sosp-replication/setup/, so the baselines dir is one level up. MP-SPDZ is
# installed into baselines/mpspdz, and this absolute path is used on every node.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "${SCRIPT_DIR}/../baselines" && pwd)"
INSTALL_DIR="${BASELINES_DIR}/mpspdz"

# MP-SPDZ's README lists these apt dependencies:
#   automake build-essential clang cmake git libboost-dev
#   libboost-filesystem-dev libboost-iostreams-dev libboost-thread-dev
#   libgmp-dev libntl-dev libsodium-dev libssl-dev libtool python3
# CryptDough's scripts/setup/_setup_required.sh already installs automake,
# build-essential, cmake, git, libsodium-dev, libtool, and python3 (with pip),
# so the list below is exactly the remainder, plus two extras required to run
# the tutorial across the cluster with MP-SPDZ's remote runner:
#   - python3-fabric:  compile-run.py -H uses Fabric/SSH to upload and launch.
#   - libstdc++-11-dev: provide the GNU C++ standard library for clang.
MISSING_DEPS=(
    clang
    libboost-dev
    libboost-filesystem-dev
    libboost-iostreams-dev
    libboost-thread-dev
    libgmp-dev
    libntl-dev
    libssl-dev
    python3-fabric
    libstdc++-11-dev
)

NPROC="${#NODES[@]}"

LOG_FILE="~/install_mpspdz.log"
echo "==> ${NPROC} node(s) supplied; installing MP-SPDZ and building the MASCOT VM."
echo "==> Each node writes its output to ${LOG_FILE} on its own host."

# Install missing deps + clone + make setup + build the VM on every node, in parallel.

remote_build_script() {
    cat <<REMOTE
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "[mpspdz] Installing MP-SPDZ build dependencies not already provided by CryptDough..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends ${MISSING_DEPS[*]}

echo "[mpspdz] Cloning ${REPO_URL} into ${INSTALL_DIR}..."
mkdir -p "$(dirname "${INSTALL_DIR}")"
if [[ ! -d "${INSTALL_DIR}/.git" ]]; then
    git clone "${REPO_URL}" "${INSTALL_DIR}"
fi
cd "${INSTALL_DIR}"
git fetch --all --tags
git checkout ${REPO_COMMIT}

echo "[mpspdz] Running 'make setup' (builds the bundled crypto dependencies)..."
make setup

echo "[mpspdz] Building the MASCOT virtual machine ('make -j mascot-party.x')..."
make -j mascot-party.x

echo "[mpspdz] Build complete."
REMOTE
}

echo "==> Phase 1/2: installing + cloning + building on ${NPROC} node(s) in parallel..."
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

# Run MP-SPDZ's simple tutorial example across the cluster from node0.
#
# HOSTS lists every node with the absolute install path. MP-SPDZ's HOSTS format
# is "<host><path>": a single "/" after the host is home-relative, while "//"
# is root-relative (absolute). INSTALL_DIR already starts with "/", so
# "${node}/${INSTALL_DIR}" yields "<node>//<abs-path>", i.e. an absolute path.

remote_run_script() {
    cat <<REMOTE
set -euo pipefail
cd "${INSTALL_DIR}"

echo "[mpspdz] Writing HOSTS file for ${NPROC} parties..."
: > HOSTS
$(for node in "${NODES[@]}"; do
    printf 'echo "%s/%s" >> HOSTS\n' "${node#*@}" "${INSTALL_DIR}"
done)

echo "[mpspdz] Providing tutorial inputs for parties 0 and 1..."
mkdir -p Player-Data
echo 1 2 3 4 > Player-Data/Input-P0-0
echo 1 2 3 4 > Player-Data/Input-P1-0

echo "[mpspdz] Generating SSL certificates for ${NPROC} parties..."
Scripts/setup-ssl.sh ${NPROC}

echo "[mpspdz] Running: Scripts/compile-run.py -H HOSTS -E mascot tutorial"
Scripts/compile-run.py -H HOSTS -E mascot tutorial
REMOTE
}

echo "==> Phase 2/2: launching the tutorial example across the cluster from node0 (${NODES[0]})..."
if ssh "${NODES[0]}" "bash -s" <<<"$(remote_run_script)"; then
    run_ok=1
else
    run_ok=0
fi

echo
echo "======================================================================"
if [[ "$run_ok" -ne 1 ]]; then
    echo "MP-SPDZ installed on all nodes, but the cluster example did not complete."
    echo "Inspect the output above and ${LOG_FILE} on each host."
    exit 1
fi
echo "SUCCESS: MP-SPDZ cloned into ${INSTALL_DIR} on all nodes, built, and the"
echo "tutorial example ran across: ${NODES[*]}"
echo "======================================================================"

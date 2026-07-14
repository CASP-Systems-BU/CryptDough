#!/usr/bin/env bash
#
# install_pigeon.sh — Install Pigeon (chart21/hpmpc) on 4 machines in parallel.
#
# For each of the 4 supplied hosts this script, over SSH and in parallel:
#   1. Installs the required apt system dependencies,
#   2. Clones the hpmpc repository and initializes all submodules recursively.
#   3. Builds the test binary with FUNCTION_IDENTIFIER=54 PROTOCOL=12.
#   4. Once every node is built, it launches the correctness test on all 4 nodes simultaneously.
#
#   make -j PARTY=$PID FUNCTION_IDENTIFIER=54 PROTOCOL=12 && \
#   scripts/run.sh -p $PID -n 4 -a $IP0 -b $IP1 -c $IP2 -d $IP3
#
# Usage:
#   ./install_pigeon.sh <host0> <host1> <host2> <host3>
#
# where each <hostN> is an IP address, "user@IP", or an SSH alias (e.g. from
# ~/.ssh/config) reachable via `ssh <hostN>`. Party N is installed on host N.
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <host0> <host1> <host2> <host3>" >&2
    echo "  Each host is an IP address or SSH alias for one of the 4 parties." >&2
    exit 1
fi

HOSTS=("$1" "$2" "$3" "$4")

REPO_URL="https://github.com/chart21/hpmpc"
REPO_DIR="hpmpc"

# IPs the parties advertise to each other over the wire. A host may be given as
# "user@IP" for SSH; strip any "user@" prefix so run.sh gets the bare address.
IP0="${HOSTS[0]#*@}"; IP1="${HOSTS[1]#*@}"; IP2="${HOSTS[2]#*@}"; IP3="${HOSTS[3]#*@}"

LOG_FILE="~/install_pigeon.log"
echo "==> Each party writes its output to ${LOG_FILE} on its own host."

# Install dependencies, clone, and build — in parallel on all nodes.

remote_build_script() {
    local pid="$1"
    cat <<REMOTE
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "[party ${pid}] Installing system dependencies..."
sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends gcc-12 g++-12 \
    libeigen3-dev libssl-dev git vim ca-certificates python3 jq bc \
    build-essential iproute2 iperf && \
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 \
    100 --slave /usr/bin/g++ g++ /usr/bin/g++-12 && \
    sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 100 && \
    sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100

echo "[party ${pid}] Cloning ${REPO_URL} (with submodules)..."
if [[ ! -d "${REPO_DIR}/.git" ]]; then
    git clone "${REPO_URL}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"
git submodule update --init --recursive

echo "[party ${pid}] Building (FUNCTION_IDENTIFIER=54 PROTOCOL=12)..."
make -j PARTY=${pid} FUNCTION_IDENTIFIER=54 PROTOCOL=12

echo "[party ${pid}] Build complete."
REMOTE
}

echo "==> Phase 1/2: installing + building on all 4 nodes in parallel..."
build_pids=()
for i in 0 1 2 3; do
    echo "    - party $i -> ${HOSTS[$i]}"
    ssh "${HOSTS[$i]}" "bash -s > ${LOG_FILE} 2>&1" \
        <<<"$(remote_build_script "$i")" &
    build_pids+=("$!")
done

build_failed=0
for i in 0 1 2 3; do
    if ! wait "${build_pids[$i]}"; then
        echo "!! Build FAILED on party ${i} (${HOSTS[$i]}). See ${LOG_FILE} on that host." >&2
        build_failed=1
    else
        echo "    party ${i} build OK"
    fi
done

if [[ "$build_failed" -ne 0 ]]; then
    echo "==> One or more builds failed; aborting before the connectivity test." >&2
    exit 1
fi
echo "==> All 4 nodes built successfully."

# Run the correctness test on all nodes at once.

remote_run_script() {
    local pid="$1"
    cat <<REMOTE
set -uo pipefail
cd "${REPO_DIR}"
pkill -9 -f run-P 2>/dev/null || true   # clear any stale party from a prior run (frees ports)
echo "[party ${pid}] Running: make -j PARTY=${pid} FUNCTION_IDENTIFIER=54 PROTOCOL=12 && scripts/run.sh -p ${pid} -n 4 -a ${IP0} -b ${IP1} -c ${IP2} -d ${IP3}"
make -j PARTY=${pid} FUNCTION_IDENTIFIER=54 PROTOCOL=12 && \
scripts/run.sh -p ${pid} -n 4 -a ${IP0} -b ${IP1} -c ${IP2} -d ${IP3}
REMOTE
}

echo "==> Phase 2/2: launching connectivity test on all 4 nodes simultaneously..."
run_pids=()
for i in 0 1 2 3; do
    ssh "${HOSTS[$i]}" "bash -s >> ${LOG_FILE} 2>&1" \
        <<<"$(remote_run_script "$i")" &
    run_pids+=("$!")
done

# The binary's teardown abort makes the exit code unreliable, so wait for each
# run to finish and then confirm the protocol actually ran by checking the log.
for i in 0 1 2 3; do
    wait "${run_pids[$i]}" || true
done

run_failed=0
for i in 0 1 2 3; do
    if ssh "${HOSTS[$i]}" "grep -q 'P${i}, ONLINE' ${LOG_FILE}"; then
        echo "    party ${i} ran and exchanged data OK"
    else
        echo "!! party ${i} (${HOSTS[$i]}) did NOT complete the protocol. See ${LOG_FILE} on that host." >&2
        run_failed=1
    fi
done

echo
echo "======================================================================"
if [[ "$run_failed" -ne 0 ]]; then
    echo "Pigeon installed, but the 4-party test did not complete on every node."
    echo "Inspect ${LOG_FILE} on each host."
    exit 1
fi
echo "SUCCESS: Pigeon installed; all 4 parties ran the test and exchanged data."
echo "Per-node output: ${LOG_FILE} on each host."
echo "======================================================================"

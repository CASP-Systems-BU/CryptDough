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
REPO_COMMIT="3d714566858739b430267c366dc9313ece0e0394"

# Resolve the install location from this script's own location so it never depends
# on the caller's working directory. The script lives in sosp-replication/setup/,
# so the baselines dir is one level up. Pigeon is installed into baselines/hpmpc,
# and this absolute path is passed to every remote node.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "${SCRIPT_DIR}/../baselines" && pwd)"
INSTALL_DIR="${BASELINES_DIR}/hpmpc"

# IPs the parties advertise to each other over the wire. A host may be given as
# "user@host" for SSH, and the host part may be a hostname or SSH alias rather
# than a bare IP. hpmpc's Connect() (core/networking/socket.hpp) resolves peers
# with inet_aton(), which accepts ONLY dotted-decimal IPv4 addresses and rejects
# hostnames. Passing a hostname makes the connecting parties throw
# "Invalid address" in the live phase and abort. So strip any "user@" prefix and
# resolve every host to its dotted-decimal IP before handing it to run.sh.
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
IP0="$(resolve_ip "${HOSTS[0]}")"; IP1="$(resolve_ip "${HOSTS[1]}")"
IP2="$(resolve_ip "${HOSTS[2]}")"; IP3="$(resolve_ip "${HOSTS[3]}")"
echo "==> Wire IPs: P0=${IP0} P1=${IP1} P2=${IP2} P3=${IP3}"

# The nodes may share a home directory over NFS, so each party writes to its own
# uniquely named log file (~/install_pigeon-P<party>.log). A single shared name
# would be clobbered by the other parties' concurrent redirects, garbling the
# logs and breaking the per-party completion check below.
LOG_FILE_TEMPLATE="~/install_pigeon-P<party>.log"
echo "==> Each party writes its output to ${LOG_FILE_TEMPLATE} on its own host."

# Install dependencies (interactive, sudo) first, then clone + build on every
# node in parallel.

# Dependency installation needs sudo, which requires a password on most nodes.
# When ssh runs a command non-interactively (no TTY) sudo cannot prompt, so this
# phase runs on its own with a forced pseudo-terminal (ssh -tt) and is executed
# sequentially per node so each password prompt reaches your terminal cleanly.
remote_deps_script() {
    cat <<REMOTE
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "[pigeon] Installing system dependencies..."
sudo apt-get update && \
    sudo apt-get install -y \
    git vim \
    libssl-dev libeigen3-dev build-essential iproute2 iperf \
    python3 \
    jq bc
REMOTE
}

remote_build_script() {
    local pid="$1"
    cat <<REMOTE
set -euo pipefail

echo "[party ${pid}] Cloning ${REPO_URL} into ${INSTALL_DIR} (with submodules)..."
mkdir -p "$(dirname "${INSTALL_DIR}")"
if [[ ! -d "${INSTALL_DIR}/.git" ]]; then
    git clone "${REPO_URL}" "${INSTALL_DIR}"
fi
cd "${INSTALL_DIR}"
git fetch --all --tags
git checkout ${REPO_COMMIT}
git submodule update --init --recursive

echo "[party ${pid}] Building (FUNCTION_IDENTIFIER=54 PROTOCOL=12)..."
make -j PARTY=${pid} FUNCTION_IDENTIFIER=54 PROTOCOL=12

echo "[party ${pid}] Build complete."
REMOTE
}

echo "==> Phase 0/2: installing system dependencies on all 4 nodes (sudo; you may be prompted for a password per node)..."
# The script is sent as a base64 argument (not via stdin) so the interactive
# terminal stays connected to sudo. If it were piped through stdin, sudo could
# neither read the password nor disable echo, so the password would be shown.
deps_b64="$(remote_deps_script | base64 | tr -d '\n')"
for i in 0 1 2 3; do
    echo "    - party $i -> ${HOSTS[$i]}"
    # -tt forces a pseudo-terminal so sudo can prompt for and read the password
    # with echo disabled; bash reads the decoded script from the pipe, leaving
    # the terminal free for the password.
    if ! ssh -tt "${HOSTS[$i]}" "echo ${deps_b64} | base64 -d | bash"; then
        echo "!! Dependency installation FAILED on party ${i} (${HOSTS[$i]})." >&2
        exit 1
    fi
    echo "    party ${i} dependencies OK"
done
echo "==> Dependencies installed on all 4 nodes."

echo "==> Phase 1/2: cloning + building on all 4 nodes in parallel..."
build_pids=()
for i in 0 1 2 3; do
    echo "    - party $i -> ${HOSTS[$i]}"
    ssh "${HOSTS[$i]}" "bash -s > ~/install_pigeon-P${i}.log 2>&1" \
        <<<"$(remote_build_script "$i")" &
    build_pids+=("$!")
done

build_failed=0
for i in 0 1 2 3; do
    if ! wait "${build_pids[$i]}"; then
        echo "!! Build FAILED on party ${i} (${HOSTS[$i]}). See ~/install_pigeon-P${i}.log on that host." >&2
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
cd "${INSTALL_DIR}"
pkill -9 -f run-P 2>/dev/null || true   # clear any stale party from a prior run (frees ports)
echo "[party ${pid}] Running: make -j PARTY=${pid} FUNCTION_IDENTIFIER=54 PROTOCOL=12 && scripts/run.sh -p ${pid} -n 4 -a ${IP0} -b ${IP1} -c ${IP2} -d ${IP3}"
make -j PARTY=${pid} FUNCTION_IDENTIFIER=54 PROTOCOL=12 && \
scripts/run.sh -p ${pid} -n 4 -a ${IP0} -b ${IP1} -c ${IP2} -d ${IP3}
REMOTE
}

echo "==> Phase 2/2: launching connectivity test on all 4 nodes simultaneously..."
run_pids=()
for i in 0 1 2 3; do
    ssh "${HOSTS[$i]}" "bash -s >> ~/install_pigeon-P${i}.log 2>&1" \
        <<<"$(remote_run_script "$i")" &
    run_pids+=("$!")
done

# Wait for each run to finish, then confirm correctness from the logs. A correct
# run prints "Passed N out of N tests." on every party and exits cleanly; the
# per-party log is the source of truth.
for i in 0 1 2 3; do
    wait "${run_pids[$i]}" || true
done

run_failed=0
for i in 0 1 2 3; do
    # Success = "Passed N out of N tests." with both counts equal (all tests
    # passed). The backreference \1 requires the second number to match the
    # first. Anything else (crash, mismatch, or missing line) is a failure.
    if ssh "${HOSTS[$i]}" "grep -q 'Passed \([0-9][0-9]*\) out of \1 tests' ~/install_pigeon-P${i}.log"; then
        echo "    party ${i} passed all correctness tests"
    else
        echo "!! party ${i} (${HOSTS[$i]}) did NOT pass all tests. See ~/install_pigeon-P${i}.log on that host." >&2
        run_failed=1
    fi
done

echo
echo "======================================================================"
if [[ "$run_failed" -ne 0 ]]; then
    echo "Pigeon installed, but the 4-party test did not pass on every node."
    echo "Inspect ~/install_pigeon-P<party>.log on each host."
    exit 1
fi
echo "SUCCESS: Pigeon installed; all 4 parties passed the correctness test."
echo "Per-node output: ~/install_pigeon-P<party>.log on each host."
echo "======================================================================"

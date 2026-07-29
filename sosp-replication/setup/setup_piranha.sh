#!/usr/bin/env bash
#
# setup_piranha.sh — Install Piranha (ucbrise/piranha) on a 2-node GPU cluster.
#
# Unlike orchestrate_piranha.sh (which additionally provisions a fresh 2-node
# AWS GPU cluster), this script assumes the two nodes ALREADY exist, each has a
# CUDA-capable NVIDIA GPU (the driver is installed automatically if missing, so
# the script works on any cluster), and node0 has SSH access to node1. Run it
# from node0. For each node, over SSH, it:
#   1. Ensures the NVIDIA driver is active: if nvidia-smi does not work yet, it
#      installs the recommended driver via `ubuntu-drivers autoinstall` and tries
#      to load it without a reboot. If the driver cannot be activated without a
#      reboot, the script stops and asks you to reboot the node(s) and re-run.
#      It then installs the apt build dependencies (git, build-essential, cmake,
#      libgtest-dev, libssl-dev, python3, gcc-10).
#   2. Clones the Piranha repository at the pinned commit into the SAME absolute
#      path on every node (sosp-replication/baselines/piranha) and initializes
#      its submodules recursively.
#   3. Builds gtest (Piranha's tests link against libgtest), creates Piranha's
#      required output/ and files/ directories, and downloads the MNIST dataset
#      used by the SecureML smoke test into a local Python virtualenv.
#   4. Installs the CUDA 11.8 toolkit + gcc-10 side-by-side and builds the
#      2-party (TWOPC) Piranha binary with them. Piranha is CUDA-11-era code: its
#      Makefile pins CUDA 11.5 and its Thrust usage does not compile against the
#      CCCL Thrust shipped in CUDA 12.4+/13, so CUDA 11.8 (the oldest CUDA that
#      still supports modern GPUs such as the L4's sm_89) is installed alongside
#      whatever CUDA the driver ships. gcc-10 is the nvcc host compiler because
#      CUDA 11.8's nvcc miscompiles GCC 11's libstdc++ <ratio>.
#
# Once both nodes have built, it runs a minimal distributed 2-party SecureML
# test across the cluster from node0 (mirroring orchestrate_piranha.sh's smoke
# test): party 0 listens on node0, party 1 connects from node1.
#
# The install path is derived from this script's own location and is identical
# on every node, matching the convention of the other baseline setup scripts
# (setup_mpspdz.sh, setup_tva.sh, setup_pigeon.sh): the CryptDough repo is
# deployed to the same absolute path on every node.
#
# Usage:
#   ./setup_piranha.sh <node0> <node1>
#
# Exactly 2 nodes must be given (Piranha runs the 2PC/TWOPC protocol here). Each
# <nodeN> is an IP address, "user@IP", or an SSH alias (e.g. from ~/.ssh/config)
# reachable via `ssh <nodeN>`. node0 (the first argument) is the main node and
# must be able to SSH into node1 so the smoke test can run across the cluster.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <node0> <node1>" >&2
    echo "  Provide exactly 2 nodes; Piranha runs the 2-party (TWOPC) protocol." >&2
    echo "  node0 (first argument) is the main node and must have SSH access to node1." >&2
    exit 1
fi

NODES=("$1" "$2")

REPO_URL="https://github.com/ucbrise/piranha.git"
REPO_COMMIT="dfbcb59d4e24ab69eb3606b49a102e602fdbee87"

# Piranha build tuning, matching orchestrate_piranha.sh: 26-bit fixed-point
# precision, the 2-party protocol, and the CUDA 11.8 toolchain used to build it.
PIRANHA_FLOAT_PRECISION=26
PIRANHA_PROTOCOL_FLAG="-DTWOPC"
CUDA_VERSION="11.8"
CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"

# Resolve the install location from this script's own location so it never
# depends on the caller's working directory. The script lives in
# sosp-replication/setup/, so the baselines dir is one level up. Piranha is
# installed into baselines/piranha, and this absolute path is used on every node.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "${SCRIPT_DIR}/../baselines" && pwd)"
INSTALL_DIR="${BASELINES_DIR}/piranha"

# IPs the parties advertise to each other over the wire. A host may be given as
# "user@host", and the host part may be an SSH alias rather than a bare IP.
# Piranha's config lists party_ips as dotted-decimal addresses, so strip any
# "user@" prefix and resolve every host to its dotted-decimal IPv4 address.
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
        exit 1
    fi
    printf '%s' "$ip"
}

# Remote login user for a node (the part before "@"), or empty if not given.
node_user() {
    local host="$1"
    if [[ "$host" == *@* ]]; then
        printf '%s' "${host%%@*}"
    fi
    return 0
}

IP0="$(resolve_ip "${NODES[0]}")"
IP1="$(resolve_ip "${NODES[1]}")"
USER0="$(node_user "${NODES[0]}")"
USER1="$(node_user "${NODES[1]}")"
echo "==> Wire IPs: party0=${IP0} (node ${NODES[0]}), party1=${IP1} (node ${NODES[1]})"

# Piranha's driver Makefile lists these apt dependencies (README): libgtest-dev,
# libssl-dev. We also need the standard build toolchain plus gcc-10/g++-10 as the
# CUDA 11.8 nvcc host compiler, python3(+venv) for the MNIST download, and wget
# to fetch the CUDA 11.8 runfile. The NVIDIA GPU driver is installed separately
# (via ubuntu-drivers) at the start of Phase 0 if it is not already active.
DEPS=(
    git
    build-essential
    cmake
    libgtest-dev
    libssl-dev
    python3
    python3-venv
    python3-pip
    gcc-10
    g++-10
    wget
)

LOG_FILE="~/install_piranha.log"
echo "==> Installing Piranha into ${INSTALL_DIR} on both nodes and building the 2PC binary."
echo "==> Each node writes its output to ${LOG_FILE} on its own host."

# ---------------------------------------------------------------------------
# Phase 0: install build dependencies on both nodes (interactive sudo).
# ---------------------------------------------------------------------------
# Dependency installation needs sudo, which requires a password on most nodes.
# When ssh runs a command non-interactively (no TTY) sudo cannot prompt, so this
# phase runs on its own with a forced pseudo-terminal (ssh -tt) and is executed
# sequentially per node so each password prompt reaches your terminal cleanly.
# All sudo-requiring work runs here so the parallel build phase below needs no
# TTY: the NVIDIA driver install, apt dependencies, the gtest library
# build/install, and the side-by-side CUDA 11.8 toolkit install.
#
# Exit status contract (consumed by the Phase 0 loop below):
#   0  = GPU driver active and all dependencies installed; ready to build.
#   90 = dependencies installed and driver installed, but the driver could not
#        be activated without a reboot; the node must be rebooted and the script
#        re-run.
#   other non-zero = hard failure (no GPU, apt error, etc.).
remote_deps_script() {
    cat <<REMOTE
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "[piranha] Updating apt and checking for an NVIDIA GPU..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends pciutils
if ! lspci | grep -iq 'NVIDIA'; then
    echo "!! No NVIDIA GPU found on this node (lspci). Piranha requires a CUDA-capable GPU." >&2
    exit 1
fi

# Ensure the NVIDIA driver is active. If nvidia-smi already works, keep it;
# otherwise install the recommended driver generically with ubuntu-drivers and
# try to load it without a reboot.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    driver_ready=1
    echo "[piranha] NVIDIA driver already active: \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    echo "[piranha] NVIDIA driver not active; installing via 'ubuntu-drivers autoinstall'..."
    sudo apt-get install -y --no-install-recommends ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall
    # Try to load the freshly installed driver so we can avoid a reboot.
    sudo modprobe nvidia 2>/dev/null || true
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        driver_ready=1
        echo "[piranha] NVIDIA driver active after install: \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    else
        driver_ready=0
        echo "[piranha] NVIDIA driver installed but not yet loaded; a reboot is required."
    fi
fi

echo "[piranha] Installing Piranha build dependencies..."
sudo apt-get install -y --no-install-recommends ${DEPS[*]}

# Build gtest: Piranha's tests link libgtest, but on Ubuntu libgtest-dev ships
# only the sources, so they must be compiled and installed to /usr/lib.
echo "[piranha] Building and installing gtest..."
if [ -d /usr/src/googletest ]; then GTEST_SRC=/usr/src/googletest; else GTEST_SRC=/usr/src/gtest; fi
sudo cmake -S "\$GTEST_SRC" -B "\$GTEST_SRC/build"
sudo cmake --build "\$GTEST_SRC/build" -j\$(nproc)
sudo find "\$GTEST_SRC/build" -name 'libgtest*.a' -exec cp -v {} /usr/lib/ \\;

# Install Piranha's required CUDA toolchain side-by-side. Piranha is CUDA-11-era
# code: its Makefile pins CUDA 11.5 and its Thrust usage does not compile against
# the CCCL Thrust in CUDA 12.4+/13, so CUDA ${CUDA_VERSION} (the oldest CUDA that still
# supports modern GPUs such as the L4's sm_89) is installed alongside whatever
# CUDA the driver ships. gcc-10 (installed above) is the nvcc host compiler.
echo "[piranha] Installing CUDA ${CUDA_VERSION} toolkit..."
if [ ! -x ${CUDA_HOME}/bin/nvcc ]; then
    wget -q -O /tmp/cuda_${CUDA_VERSION}.run \\
        https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}.0/local_installers/cuda_${CUDA_VERSION}.0_520.61.05_linux.run
    sudo sh /tmp/cuda_${CUDA_VERSION}.run --silent --toolkit --toolkitpath=${CUDA_HOME} --override
fi

# If the driver was installed but is not yet loaded, signal that a reboot is
# required so the caller can prompt for it (exit code 90).
if [ "\$driver_ready" -ne 1 ]; then
    echo "[piranha] Dependencies installed. REBOOT required to activate the NVIDIA driver." >&2
    exit 90
fi
REMOTE
}

echo "==> Phase 0/2: installing GPU driver + build dependencies on both nodes (sudo; you may be prompted for a password per node)..."
# The script is sent as a base64 argument (not via stdin) so the interactive
# terminal stays connected to sudo. If it were piped through stdin, sudo could
# neither read the password nor disable echo, so the password would be shown.
#
# Every node is processed even if some need a reboot, so a single reboot round
# covers the whole cluster. A node exits 90 when its driver was installed but
# needs a reboot to activate; any other non-zero exit is a hard failure.
reboot_needed_nodes=()
deps_failed=0
for i in "${!NODES[@]}"; do
    echo "    - node $i -> ${NODES[$i]}"
    deps_b64="$(remote_deps_script | base64 | tr -d '\n')"
    # -tt forces a pseudo-terminal so sudo can prompt for and read the password
    # with echo disabled; bash reads the decoded script from the pipe, leaving
    # the terminal free for the password.
    set +e
    ssh -tt "${NODES[$i]}" "echo ${deps_b64} | base64 -d | bash"
    status=$?
    set -e
    case "$status" in
        0)
            echo "    node ${i} ready (driver active, dependencies installed)"
            ;;
        90)
            echo "    node ${i}: driver installed, REBOOT required"
            reboot_needed_nodes+=("${NODES[$i]}")
            ;;
        *)
            echo "!! Dependency installation FAILED on node ${i} (${NODES[$i]}) (status ${status})." >&2
            deps_failed=1
            ;;
    esac
done

if [[ "$deps_failed" -ne 0 ]]; then
    echo "==> Dependency installation failed on at least one node; aborting." >&2
    exit 1
fi

if [[ "${#reboot_needed_nodes[@]}" -gt 0 ]]; then
    echo
    echo "======================================================================"
    echo "ACTION REQUIRED: the NVIDIA driver was installed but is not yet active on:"
    for n in "${reboot_needed_nodes[@]}"; do echo "    - $n"; done
    echo
    echo "Reboot the node(s) above, then re-run this script to continue. All other"
    echo "dependencies are already installed, so the re-run proceeds to the build."
    echo "  e.g.  ssh <node> 'sudo reboot'"
    echo "======================================================================"
    exit 2
fi
echo "==> Dependencies installed and the NVIDIA driver is active on both nodes."

# ---------------------------------------------------------------------------
# Phase 1: clone + build Piranha on both nodes, in parallel.
# ---------------------------------------------------------------------------
# The parallel build phase needs no sudo: all privileged steps ran in Phase 0.
remote_build_script() {
    cat <<REMOTE
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "===== Piranha build started at \$(date) ====="

# --- Clone Piranha at the pinned commit into the shared install path ---
echo "[piranha] Cloning ${REPO_URL} into ${INSTALL_DIR}..."
mkdir -p "$(dirname "${INSTALL_DIR}")"
if [[ ! -d "${INSTALL_DIR}/.git" ]]; then
    git clone "${REPO_URL}" "${INSTALL_DIR}"
fi
cd "${INSTALL_DIR}"
git fetch --all --tags
git checkout ${REPO_COMMIT}
git submodule update --init --recursive

# --- Required directories ---
echo "[piranha] Creating output/data directories..."
mkdir -p "${INSTALL_DIR}/output" "${INSTALL_DIR}/files/MNIST" "${INSTALL_DIR}/files/CIFAR10"

# --- Download MNIST (needed by the SecureML smoke test). Use a local venv with
#     CPU-only torch/torchvision so the download works on PEP 668 nodes and
#     avoids the large CUDA wheels; the dataset is only fetched in cleartext. ---
echo "[piranha] Downloading MNIST dataset (via a local .env virtualenv)..."
if [[ ! -d "${INSTALL_DIR}/.env" ]]; then
    python3 -m venv "${INSTALL_DIR}/.env"
fi
"${INSTALL_DIR}/.env/bin/pip" install --quiet --upgrade pip
"${INSTALL_DIR}/.env/bin/pip" install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
cd "${INSTALL_DIR}/scripts"
"${INSTALL_DIR}/.env/bin/python" download_mnist.py

# --- Build Piranha (2-party protocol) with the CUDA ${CUDA_VERSION} + gcc-10 toolchain
#     installed in Phase 0. Piranha uses CUTLASS headers only (no libcutlass.a),
#     so no separate CUTLASS build is needed — the submodule checkout above
#     provides the headers. ---
echo "[piranha] Building Piranha (FLOAT_PRECISION=${PIRANHA_FLOAT_PRECISION}, ${PIRANHA_PROTOCOL_FLAG})..."
cd "${INSTALL_DIR}"
export PATH=${CUDA_HOME}/bin:\$PATH
export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-10'
make -j\$(nproc) CUDA_VERSION=${CUDA_VERSION} PIRANHA_FLAGS="-DFLOAT_PRECISION=${PIRANHA_FLOAT_PRECISION} ${PIRANHA_PROTOCOL_FLAG}"

test -x "${INSTALL_DIR}/piranha"
echo "===== PIRANHA_BUILD_DONE at \$(date) ====="
REMOTE
}

echo "==> Phase 1/2: cloning + building Piranha on both nodes in parallel..."
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
        echo "!! Build FAILED on node ${i} (${NODES[$i]}). Last log lines:" >&2
        ssh "${NODES[$i]}" "tail -n 30 ${LOG_FILE}" 2>/dev/null || true
        build_failed=1
    else
        echo "    node ${i} build OK"
    fi
done

if [[ "$build_failed" -ne 0 ]]; then
    echo "==> One or more builds failed; aborting before the smoke test." >&2
    exit 1
fi
echo "==> Both nodes built successfully."

# ---------------------------------------------------------------------------
# Phase 2: run a minimal distributed 2-party SecureML test across the cluster.
# ---------------------------------------------------------------------------
# The piranha binary is linked against the side-by-side CUDA runtime, which is
# not on the default loader path, so every invocation needs LD_LIBRARY_PATH.
PIRANHA_RUN_ENV="LD_LIBRARY_PATH=${CUDA_HOME}/lib64 CUDA_VISIBLE_DEVICES=0"

echo "==> Phase 2/2: running the distributed 2-party smoke test from node0 (${NODES[0]})..."

# Generate the distributed 2-party config on node0 from the localhost sample,
# then copy it to node1 so both parties use identical settings.
echo "    Generating files/samples/dist_config.json on node0..."
if ! ssh "${NODES[0]}" "cd '${INSTALL_DIR}' && python3 - '${IP0}' '${IP1}' '${USER0}' '${USER1}' <<'PYEOF'
import getpass, json, sys
ip0, ip1, user0, user1 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
default_user = getpass.getuser()
cfg = json.load(open('files/samples/localhost_config.json'))
cfg['num_parties'] = 2
cfg['party_ips']   = [ip0, ip1]
cfg['party_users'] = [user0 or default_user, user1 or default_user]
cfg['run_name']    = 'piranha_dist_test'
# Minimal fast smoke test: 1 epoch, 1 training iteration, no test pass.
cfg['lr_schedule']            = [3]
cfg['custom_epochs']          = True
cfg['custom_epoch_count']     = 1
cfg['custom_iterations']      = True
cfg['custom_iteration_count'] = 1
cfg['no_test']                = True
cfg['last_test']              = False
json.dump(cfg, open('files/samples/dist_config.json', 'w'), indent=2)
print('wrote files/samples/dist_config.json')
PYEOF"; then
    echo "!! Failed to generate the distributed config on node0." >&2
    exit 1
fi

echo "    Distributing config to node1..."
ssh "${NODES[0]}" "cat '${INSTALL_DIR}/files/samples/dist_config.json'" | \
    ssh "${NODES[1]}" "cat > '${INSTALL_DIR}/files/samples/dist_config.json'"

# Start party 0 (the listener) first, detached, and wait until it is actually
# listening before starting party 1 (the connector). Otherwise party 1 can
# exhaust its connection-retry budget while party 0 is still initializing CUDA,
# and the two parties never connect.
echo "    Starting party 0 on node0 (listener; detached)..."
ssh "${NODES[0]}" "echo '===== Piranha 2PC run (party 0) started at '\$(date)' =====' >> ${LOG_FILE}"
ssh -f -n "${NODES[0]}" \
    "cd '${INSTALL_DIR}' && env ${PIRANHA_RUN_ENV} ./piranha -p 0 -c files/samples/dist_config.json < /dev/null >> ${LOG_FILE} 2>&1"

echo "    Waiting for party 0 to begin listening..."
for _ in $(seq 1 30); do
    if ssh "${NODES[0]}" "ss -tln 2>/dev/null | grep -q ':3200'"; then
        echo "    party 0 is listening"
        break
    fi
    sleep 2
done

echo "    Starting party 1 on node1 (connector; foreground, streaming output)..."
set +e
ssh "${NODES[1]}" "cd '${INSTALL_DIR}' && echo '===== Piranha 2PC run (party 1) started at '\$(date)' =====' | tee -a ${LOG_FILE} && \
    env ${PIRANHA_RUN_ENV} ./piranha -p 1 -c files/samples/dist_config.json 2>&1 | tee -a ${LOG_FILE}"
run_status=$?
set -e

echo
echo "======================================================================"
if [[ "$run_status" -ne 0 ]]; then
    echo "Piranha installed and built on both nodes, but the 2-party smoke test"
    echo "did not complete (party 1 exited with status ${run_status})."
    echo "Inspect ${LOG_FILE} on each node:"
    echo "    node0: ssh ${NODES[0]} 'tail -n 40 ${LOG_FILE}'"
    echo "    node1: ssh ${NODES[1]} 'tail -n 40 ${LOG_FILE}'"
    exit 1
fi
echo "SUCCESS: Piranha cloned into ${INSTALL_DIR} on both nodes, built (2PC), and"
echo "the distributed 2-party smoke test ran across: ${NODES[*]}"
echo "Per-node output: ${LOG_FILE} on each host."
echo "======================================================================"

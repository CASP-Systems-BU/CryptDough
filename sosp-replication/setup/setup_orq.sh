#!/usr/bin/env bash
#
# setup_orq.sh — Install ORQ (CASP-Systems-BU/orq) on an MPC cluster.
#
# ORQ ships its own multi-node deployment script,
# scripts/orchestration/deploy.sh, which (per ORQ's README):
#   1. generates/copies an SSH key to every node,
#   2. copies the repository to each worker node,
#   3. installs dependencies (via setup.sh) and builds test_primitives,
#   4. runs an MPI smoke test across all nodes.
#
# This wrapper therefore needs to:
#   1. Clone the pinned ORQ commit into sosp-replication/baselines/orq.
#   2. Patch ORQ's deploy.sh so it copies the repo to the SAME absolute path on
#      every worker (baselines/orq) instead of each worker's $HOME. Upstream
#      deploy.sh does `scp -r $REPO_NAME/ $W:~/`, which lands the files in
#      ~/orq while the rest of the script references the absolute $REPO_NAME;
#      the two only agree when the repo lives directly under $HOME. We want ORQ
#      under baselines/orq on all nodes, so we redirect that copy.
#   3. Invoke ORQ's deploy.sh with the given node names.
#
# Usage:
#   ./setup_orq.sh <node0> [node1 ...]
#
# Any number of nodes may be given; ORQ's deploy.sh is parameterized by the
# node count (it sets PROTOCOL and the MPI process count to the number of
# nodes). Each <nodeN> is an IP address, "user@IP", or an SSH alias (e.g. from
# ~/.ssh/config) reachable via `ssh <nodeN>`. node0 (the first argument) is the
# main node and must be able to SSH into the others.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <node0> [node1 ...]" >&2
    echo "  node0 (first argument) is the main node and must have SSH access to the others." >&2
    exit 1
fi

NODES=("$@")

REPO_URL="https://github.com/CASP-Systems-BU/orq"
REPO_COMMIT="2d7946a95f6d1d49e020789b70a6cfbdc1198a46"

# Resolve the install location from this script's own location so it never
# depends on the caller's working directory. The script lives in
# sosp-replication/setup/, so the baselines dir is one level up. ORQ is
# installed into baselines/orq.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "${SCRIPT_DIR}/../baselines" && pwd)"
INSTALL_DIR="${BASELINES_DIR}/orq"

echo "==> Cloning ORQ into ${INSTALL_DIR} (commit ${REPO_COMMIT})..."
mkdir -p "${BASELINES_DIR}"
if [[ ! -d "${INSTALL_DIR}/.git" ]]; then
    git clone "${REPO_URL}" "${INSTALL_DIR}"
fi
cd "${INSTALL_DIR}"
git fetch --all --tags
git checkout "${REPO_COMMIT}"
git submodule update --init --recursive

# Patch ORQ's deploy.sh so worker copies land at ${INSTALL_DIR} (the absolute
# baselines path) on every node, instead of each worker's $HOME. Start from a
# pristine copy so the patch is deterministic across re-runs, then rewrite only
# the copy destination in the per-worker setup loop.
DEPLOY_SCRIPT="${INSTALL_DIR}/scripts/orchestration/deploy.sh"
echo "==> Patching ${DEPLOY_SCRIPT} to deploy ORQ under ${INSTALL_DIR} on all nodes..."
git checkout -- scripts/orchestration/deploy.sh
if grep -qF 'scp -r $REPO_NAME/ $W:~/' "${DEPLOY_SCRIPT}"; then
    sed -i 's|scp -r $REPO_NAME/ $W:~/|ssh $W mkdir -p "$(dirname "$REPO_NAME")"; scp -r $REPO_NAME $W:$REPO_NAME|' "${DEPLOY_SCRIPT}"
fi
if ! grep -qF 'scp -r $REPO_NAME $W:$REPO_NAME' "${DEPLOY_SCRIPT}"; then
    echo "!! Failed to patch ${DEPLOY_SCRIPT}: expected copy line not found." >&2
    echo "!! Upstream deploy.sh may have changed; aborting before deploy." >&2
    exit 1
fi

echo "==> Deploying ORQ to the cluster via ORQ's deploy.sh..."
echo "    ${INSTALL_DIR}/scripts/orchestration/deploy.sh ${INSTALL_DIR} ${NODES[*]}"
"${INSTALL_DIR}/scripts/orchestration/deploy.sh" "${INSTALL_DIR}" "${NODES[@]}"

echo
echo "======================================================================"
echo "SUCCESS: ORQ cloned into ${INSTALL_DIR} and deployed to: ${NODES[*]}"
echo "======================================================================"

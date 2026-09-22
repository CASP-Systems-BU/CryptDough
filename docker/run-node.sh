#!/usr/bin/env bash
#
# Start the CryptDough container on one cluster machine.
#
# Run this on EVERY machine that will host a party, with the SAME --nodes list.
# The list is positional: the first entry becomes node0, the second node1, etc.
#
#   ./docker/run-node.sh --nodes blinky,pinky,inky,clyde
#
# Then drive the experiment from the node0 machine:
#
#   docker exec -it -u cdough cdough ../scripts/run_experiment.py -s lan -c nocopy -n 4 -T 8 micro_sorting
#
# Note the `-u cdough`: `docker exec` bypasses the image's ENTRYPOINT, which is what
# normally drops privileges, so without it the command runs as root.
#
# Why the --nodes list matters: run_experiment.py addresses peers as <prefix><i>
# (default prefix "node") and assumes consistent numbering, but real machines are
# rarely named node0..node3. Rather than requiring root to edit /etc/hosts on the
# host, this script resolves each machine to an IP and injects node0..nodeN-1
# aliases into the container's own /etc/hosts via --add-host.

set -euo pipefail

IMAGE="${IMAGE:-cryptdough:latest}"
NAME="${NAME:-cdough}"
NODES=""
KEY_DIR="${KEY_DIR:-$HOME/.cdough-cluster}"
BUILD_VOLUME="${BUILD_VOLUME:-cdough-build}"
SRC_MOUNT=""
WAN=0
EXTRA_ARGS=()

usage() {
    cat <<'USAGE'
Usage: run-node.sh --nodes h0,h1[,h2,h3] [options]

Required:
  --nodes LIST      Comma-separated machine list, in party order.
                    The first entry becomes node0, the second node1, ...

Options:
  --image IMG       Image to run                     (default: cryptdough:latest)
  --name NAME       Container name                   (default: cdough)
  --key-dir DIR     Directory holding id_ed25519 [+ .pub]
                                                     (default: ~/.cdough-cluster)
  --src PATH        Bind-mount a host source tree over /opt/cryptdough
                    (for iterating on code without rebuilding the image)
  --build-volume V  Named volume for the build dir   (default: cdough-build)
  --wan             Add --cap-add NET_ADMIN, needed for `-s wan` (tc qdisc)
  --                Everything after this is passed straight to `docker run`
  -h, --help        This message
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)        NODES="$2"; shift 2 ;;
        --image)        IMAGE="$2"; shift 2 ;;
        --name)         NAME="$2"; shift 2 ;;
        --key-dir)      KEY_DIR="$2"; shift 2 ;;
        --src)          SRC_MOUNT="$2"; shift 2 ;;
        --build-volume) BUILD_VOLUME="$2"; shift 2 ;;
        --wan)          WAN=1; shift ;;
        --)             shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help)      usage; exit 0 ;;
        *)              echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "${NODES}" ]]; then
    echo "ERROR: --nodes is required" >&2
    usage
    exit 1
fi

if [[ ! -f "${KEY_DIR}/id_ed25519" ]]; then
    echo "ERROR: no keypair at ${KEY_DIR}/id_ed25519" >&2
    echo "Generate one ONCE, then copy the whole directory to every machine:" >&2
    echo "  mkdir -p ${KEY_DIR} && ssh-keygen -t ed25519 -N '' -f ${KEY_DIR}/id_ed25519" >&2
    exit 1
fi

# --- Resolve peers and build the node0..nodeN-1 alias list -------------------
IFS=',' read -r -a NODE_LIST <<< "${NODES}"
HOST_ARGS=()
rank=0
for machine in "${NODE_LIST[@]}"; do
    ip="$(getent ahostsv4 "${machine}" | awk 'NR==1 {print $1}')"
    if [[ -z "${ip}" ]]; then
        echo "ERROR: could not resolve '${machine}' to an IPv4 address" >&2
        exit 1
    fi
    HOST_ARGS+=(--add-host "node${rank}:${ip}")
    echo "  node${rank} -> ${machine} (${ip})"
    rank=$((rank + 1))
done

CAP_ARGS=()
if [[ "${WAN}" -eq 1 ]]; then
    CAP_ARGS+=(--cap-add NET_ADMIN)
fi

SRC_ARGS=()
if [[ -n "${SRC_MOUNT}" ]]; then
    SRC_ARGS+=(-v "$(cd "${SRC_MOUNT}" && pwd):/opt/cryptdough")
    echo "  source bind-mounted from ${SRC_MOUNT} (deps at /opt/cdough-deps are unaffected)"
fi

docker rm -f "${NAME}" >/dev/null 2>&1 || true

# --- port 2222 is a host-wide resource under --network host -----------------
# Only one CryptDough container can run per machine. A second one starts fine but
# its sshd cannot bind, so it is silently unreachable; startmpc and scp then talk
# to whichever container *does* own 2222, and the run fails much later with a
# confusing "stdbuf: failed to run command './<binary>': No such file or directory".
# Fail loudly here instead.
if ss -tln 2>/dev/null | grep -qE '[:.]2222[[:space:]]'; then
    holder=""
    for c in $(docker ps --format '{{.Names}}'); do
        if docker exec "$c" pgrep -x sshd >/dev/null 2>&1; then holder="$c"; break; fi
    done
    echo "ERROR: port 2222 on this host is already in use${holder:+ by container '${holder}'}." >&2
    echo "       Under --network host only one CryptDough container can run per machine." >&2
    echo "       Stop the other one first:  docker rm -f ${holder:-<container>}" >&2
    exit 1
fi

echo "Starting ${NAME} from ${IMAGE}..."
docker run -d \
    --name "${NAME}" \
    --network host \
    `# host networking: no NAT, no veth pair, native throughput, and peers` \
    `# reach this container on the host's own IP` \
    --shm-size=1g \
    `# Docker defaults /dev/shm to 64MB; OpenMPI's shared-memory transport` \
    `# exhausts that during the local MPI half of the test suite` \
    --ulimit nofile=65536:65536 \
    `# startmpc opens host_count^2 * threads sockets per party` \
    "${CAP_ARGS[@]}" \
    "${HOST_ARGS[@]}" \
    -e CDOUGH_SSHD=1 \
    -v "${KEY_DIR}:/keys:ro" \
    -v "${BUILD_VOLUME}:/opt/cryptdough/build" \
    "${SRC_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    "${IMAGE}" \
    sleep infinity

echo
echo "Container '${NAME}' is up. Verify with:"
echo "  docker exec -u cdough ${NAME} hostname"
echo "  docker exec -u cdough ${NAME} ssh node1 hostname   # from the node0 machine"
echo
echo "Run an experiment from the node0 machine:"
echo "  docker exec -it -u cdough ${NAME} ../scripts/run_experiment.py -s lan -c nocopy -n 4 -T 8 micro_sorting"

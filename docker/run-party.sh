#!/usr/bin/env bash
#
# SSH-free launcher: run ONE party directly, with no startmpc and no sshd.
#
# startmpc's only job is to pick a base port and set five environment variables
# before exec'ing the binary (see include/backend/nocopy_communicator/startmpc/startmpc
# and startmpc.h:77-113). Setting them ourselves removes the need for inter-container
# SSH entirely.
#
# Build once on each machine:
#   docker exec cdough bash -c 'cmake .. -DPROTOCOL=3 -DCOMM=NOCOPY -DCOMM_THREADS=4 && make -j micro_sorting'
#
# Then, on machine i of a 3-party run (all agreeing on --base-port):
#   ./docker/run-party.sh --rank 0 --hosts node0,node1,node2 --base-port 20000 \
#       -- ./micro_sorting -s lan -t 8 -b -12 -r 1048576

set -euo pipefail

IMAGE="${IMAGE:-cryptdough:latest}"
RANK=""
HOSTS=""
BASE_PORT="${BASE_PORT:-20000}"
BUILD_VOLUME="${BUILD_VOLUME:-cdough-build}"
CMD=()

usage() {
    cat <<'USAGE'
Usage: run-party.sh --rank N --hosts h0,h1[,h2,h3] [--base-port P] -- <command>

Required:
  --rank N          This party's rank, 0-based. Must be unique per machine.
  --hosts LIST      Comma-separated peer list in party order, as resolvable
                    from inside the container (e.g. node0,node1,node2).
  --                Separator; everything after it is the command to run.

Options:
  --base-port P     First port of the block used by the run   (default: 20000)
                    All parties MUST agree. The run uses ports
                    P .. P + host_count^2 * threads.
  --image IMG       Image to run                              (default: cryptdough:latest)
  --build-volume V  Named volume holding the compiled binary  (default: cdough-build)
  -h, --help        This message
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank)         RANK="$2"; shift 2 ;;
        --hosts)        HOSTS="$2"; shift 2 ;;
        --base-port)    BASE_PORT="$2"; shift 2 ;;
        --image)        IMAGE="$2"; shift 2 ;;
        --build-volume) BUILD_VOLUME="$2"; shift 2 ;;
        --)             shift; CMD=("$@"); break ;;
        -h|--help)      usage; exit 0 ;;
        *)              echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "${RANK}" || -z "${HOSTS}" || ${#CMD[@]} -eq 0 ]]; then
    echo "ERROR: --rank, --hosts and a command after -- are all required" >&2
    usage
    exit 1
fi

IFS=',' read -r -a HOST_LIST <<< "${HOSTS}"
HOST_COUNT="${#HOST_LIST[@]}"

if (( RANK < 0 || RANK >= HOST_COUNT )); then
    echo "ERROR: --rank ${RANK} is out of range for ${HOST_COUNT} hosts" >&2
    exit 1
fi

echo "party ${RANK}/${HOST_COUNT}  hosts=${HOSTS}  base_port=${BASE_PORT}"

# -t only when there is a TTY: this script is normally invoked over ssh from a
# driver script, where `docker run -it` would fail with "the input device is not a TTY".
TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
    TTY_ARGS+=(-it)
fi

exec docker run --rm "${TTY_ARGS[@]}" \
    --network host \
    --shm-size=1g \
    --ulimit nofile=65536:65536 \
    -e STARTMPC_EXEC_MODE=1 \
    -e STARTMPC_HOST_RANK="${RANK}" \
    -e STARTMPC_HOST_COUNT="${HOST_COUNT}" \
    -e STARTMPC_HOST_LIST="${HOSTS}" \
    -e STARTMPC_BASE_PORT="${BASE_PORT}" \
    -v "${BUILD_VOLUME}:/opt/cryptdough/build" \
    "${IMAGE}" \
    "${CMD[@]}"

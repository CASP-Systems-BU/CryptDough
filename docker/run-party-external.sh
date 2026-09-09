#!/usr/bin/env bash
#
# Launch THIS organization's party in a cross-organizational CryptDough run.
#
# No SSH to or from any other party. `startmpc` is bypassed entirely: it only ever
# picked a base port and set five environment variables before exec'ing the binary
# (see include/backend/nocopy_communicator/startmpc/startmpc.h), and here the parties
# agree those values in advance via the run manifest.
#
#   ./docker/run-party-external.sh \
#       --rank 0 --hosts mpc-a.example.edu,mpc-b.example.org,mpc-c.example.com \
#       --base-port 20000 --tls-dir /scratch/adam/tls --data-dir /srv/private \
#       -- ./micro_primitives -f /run/manifest.args
#
# LAUNCH ORDER MATTERS. Party i connects to every j > i and listens for every j < i.
# Listeners block in accept() indefinitely; only connectors time out. So start in
# DESCENDING rank order -- the highest rank first, rank 0 last. Higher ranks may start
# minutes early at no cost.

set -euo pipefail

IMAGE="${IMAGE:-cryptdough:latest}"
RANK=""
HOSTS=""
BASE_PORT=""
TLS_DIR=""
DATA_DIR=""
BUILD_VOLUME="${BUILD_VOLUME:-cdough-build}"
CONNECT_RETRIES="${CONNECT_RETRIES:-300}"
ATTEMPTS="${ATTEMPTS:-3}"
CMD=()

usage() {
    cat <<'USAGE'
Usage: run-party-external.sh --rank N --hosts h0,h1,h2 --base-port P [options] -- <command>

Required:
  --rank N          This party's rank (0-based), as assigned in the manifest.
  --hosts LIST      All parties in RANK ORDER, as hostnames or IPs resolvable from
                    this machine. Must be byte-identical across all parties.
  --base-port P     Agreed base port. All parties must use the same value.
  --                Separator; everything after it is the command to run.

Options:
  --tls-dir DIR     Directory with this party's key/cert and the peer certs.
                    Enables TLS. Expects party<RANK>.key, party<RANK>.crt, and the
                    other parties' party<N>.crt files.
  --data-dir DIR    This party's private data, mounted read-only at /data.
                    Never leaves this machine: only the owning party opens it.
  --image IMG       Image to run (default: cryptdough:latest).
  --build-volume V  Named volume holding the compiled binary (default: cdough-build).
  --attempts N      Relaunch this many times if the peers are not up yet (default: 3).
  --retries N       Per-connection connect retries, ~1s apart (default: 300).
  -h, --help        This message.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank)         RANK="$2"; shift 2 ;;
        --hosts)        HOSTS="$2"; shift 2 ;;
        --base-port)    BASE_PORT="$2"; shift 2 ;;
        --tls-dir)      TLS_DIR="$2"; shift 2 ;;
        --data-dir)     DATA_DIR="$2"; shift 2 ;;
        --image)        IMAGE="$2"; shift 2 ;;
        --build-volume) BUILD_VOLUME="$2"; shift 2 ;;
        --attempts)     ATTEMPTS="$2"; shift 2 ;;
        --retries)      CONNECT_RETRIES="$2"; shift 2 ;;
        --)             shift; CMD=("$@"); break ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "${RANK}" || -z "${HOSTS}" || -z "${BASE_PORT}" || ${#CMD[@]} -eq 0 ]]; then
    echo "ERROR: --rank, --hosts, --base-port and a command after -- are all required" >&2
    usage
    exit 1
fi

IFS=',' read -r -a HOST_LIST <<< "${HOSTS}"
HOST_COUNT="${#HOST_LIST[@]}"

if (( RANK < 0 || RANK >= HOST_COUNT )); then
    echo "ERROR: --rank ${RANK} is out of range for ${HOST_COUNT} hosts" >&2
    exit 1
fi

# --- TLS material -----------------------------------------------------------
TLS_ARGS=()
if [[ -n "${TLS_DIR}" ]]; then
    own_key="${TLS_DIR}/party${RANK}.key"
    own_crt="${TLS_DIR}/party${RANK}.crt"
    for f in "${own_key}" "${own_crt}"; do
        [[ -f "$f" ]] || { echo "ERROR: missing ${f}. Run docker/gen-party-certs.sh." >&2; exit 1; }
    done

    # Pin every OTHER party's certificate. Our own is excluded: a peer presenting it
    # would mean someone is replaying our identity back at us.
    peer_certs=()
    for ((r = 0; r < HOST_COUNT; r++)); do
        (( r == RANK )) && continue
        crt="${TLS_DIR}/party${r}.crt"
        [[ -f "${crt}" ]] || { echo "ERROR: missing peer certificate ${crt}." >&2; exit 1; }
        peer_certs+=("/tls/party${r}.crt")
    done
    peer_list="$(IFS=','; echo "${peer_certs[*]}")"

    TLS_ARGS+=(
        -v "$(cd "${TLS_DIR}" && pwd):/tls:ro"
        -e "CDOUGH_TLS_CERT=/tls/party${RANK}.crt"
        -e "CDOUGH_TLS_KEY=/tls/party${RANK}.key"
        -e "CDOUGH_TLS_PEER_CERTS=${peer_list}"
    )
    echo "TLS enabled; pinning ${#peer_certs[@]} peer certificate(s)"
else
    echo "WARNING: no --tls-dir given. Traffic will be UNENCRYPTED and the PRG seed" >&2
    echo "         exchange will be readable by anyone on the path. Only acceptable" >&2
    echo "         on a network you fully trust." >&2
fi

DATA_ARGS=()
if [[ -n "${DATA_DIR}" ]]; then
    DATA_ARGS+=(-v "$(cd "${DATA_DIR}" && pwd):/data:ro")
fi

# --- Report the ports this party needs open ---------------------------------
if (( RANK == 0 )); then
    echo "Inbound ports required: none (rank 0 only makes outbound connections)"
else
    echo -n "Inbound ports required: "
    for ((j = 0; j < RANK; j++)); do
        # Mirrors startmpc.h: base + host_count*threads*j + threads*rank.
        # Reported per-thread-block; --hosts fixes host_count, threads comes from -t.
        echo -n "(from rank ${j}) "
    done
    echo "- see docker/check-manifest.sh for exact ranges"
fi

echo "party ${RANK}/${HOST_COUNT}  hosts=${HOSTS}  base_port=${BASE_PORT}"

TTY_ARGS=()
[[ -t 0 && -t 1 ]] && TTY_ARGS+=(-it)

# --- Retry supervisor -------------------------------------------------------
# Peers are separate organizations launching independently, so this party may come up
# before them. The in-process connect retry (CDOUGH_CONNECT_RETRIES) covers ordinary
# skew; this outer loop covers a peer that has not started at all yet.
attempt=1
while :; do
    echo "=== attempt ${attempt}/${ATTEMPTS} at $(date '+%Y-%m-%d %H:%M:%S') ==="
    set +e
    docker run --rm "${TTY_ARGS[@]}" \
        --network host \
        --shm-size=1g \
        --ulimit nofile=65536:65536 \
        -e STARTMPC_EXEC_MODE=1 \
        -e STARTMPC_HOST_RANK="${RANK}" \
        -e STARTMPC_HOST_COUNT="${HOST_COUNT}" \
        -e STARTMPC_HOST_LIST="${HOSTS}" \
        -e STARTMPC_BASE_PORT="${BASE_PORT}" \
        -e CDOUGH_CONNECT_RETRIES="${CONNECT_RETRIES}" \
        "${TLS_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        -v "${BUILD_VOLUME}:/opt/cryptdough/build" \
        "${IMAGE}" \
        "${CMD[@]}"
    status=$?
    set -e

    if [[ ${status} -eq 0 ]]; then
        echo "=== party ${RANK} completed successfully ==="
        exit 0
    fi

    if (( attempt >= ATTEMPTS )); then
        echo "=== party ${RANK} failed after ${ATTEMPTS} attempts (exit ${status}) ===" >&2
        exit "${status}"
    fi
    echo "exit ${status}; retrying in 10s (are the higher-ranked parties up?)" >&2
    sleep 10
    attempt=$((attempt + 1))
done

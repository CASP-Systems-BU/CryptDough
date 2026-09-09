#!/usr/bin/env bash
#
# Build this organization's CryptDough image from the agreed commit and manifest.
#
# Every party builds its own image. No organization runs a binary another organization
# compiled -- that is the only defensible arrangement when the parties do not trust
# each other, and it costs each of them one build.
#
#   ./docker/build-party.sh --manifest run.yaml --src /path/to/CryptDough
#
# Prints a fingerprint of {commit, compile-time options} for the parties to compare out
# of band. Note that images are NOT bit-reproducible: CMakeLists.txt builds with
# -march=native, so each party's binary is tuned to its own CPU. Verification is at the
# source-and-configuration level, which is what the fingerprint captures.

set -euo pipefail

MANIFEST=""
SRC="${SRC:-.}"
TAG="${TAG:-cryptdough:latest}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"

usage() {
    cat <<'USAGE'
Usage: build-party.sh --manifest FILE [--src DIR] [--tag TAG] [--jobs N]
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --src)      SRC="$2"; shift 2 ;;
        --tag)      TAG="$2"; shift 2 ;;
        --jobs)     JOBS="$2"; shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

[[ -n "${MANIFEST}" ]] || { echo "ERROR: --manifest is required" >&2; usage; exit 1; }
[[ -f "${MANIFEST}" ]] || { echo "ERROR: no such manifest: ${MANIFEST}" >&2; exit 1; }

value_of() {
    sed -n "s/^[[:space:]]*$1:[[:space:]]*//p" "${MANIFEST}" \
        | head -1 | sed 's/[[:space:]]*#.*$//' | tr -d '"'"'"''
}

want_commit="$(value_of git_commit)"
have_commit="$(git -C "${SRC}" rev-parse HEAD 2>/dev/null || echo unknown)"

if [[ -n "${want_commit}" && "${want_commit}" != 0000000000000000000000000000000000000000 ]]; then
    if [[ "${have_commit}" != "${want_commit}" ]]; then
        echo "ERROR: checked out ${have_commit}, manifest requires ${want_commit}" >&2
        echo "       git -C ${SRC} checkout ${want_commit}" >&2
        exit 1
    fi
fi
if [[ -n "$(git -C "${SRC}" status --porcelain 2>/dev/null)" ]]; then
    echo "ERROR: working tree is dirty; the other parties would be building something else." >&2
    exit 1
fi

echo "Building ${TAG} from ${have_commit:0:12}"
# --no-cache is deliberate: the legacy (non-BuildKit) builder does not invalidate
# COPY --from=<stage> when the source stage changes, so an incremental build can
# silently ship stale dependencies. See docker/README.md.
docker build --no-cache --build-arg BUILD_JOBS="${JOBS}" -t "${TAG}" -f "${SRC}/docker/Dockerfile" "${SRC}"

# --- fingerprint ------------------------------------------------------------
opts=""
for key in PROTOCOL COMM COMM_THREADS TLS TRIPLES DEFAULT_BITWIDTH \
           SORT_PROTO BOOLEAN_ADDER DIVISION_CORRECTION; do
    opts+="${key}=$(value_of "${key}");"
done
fingerprint="$(printf '%s|%s' "${have_commit}" "${opts}" | sha256sum | cut -d' ' -f1)"

cat <<SUMMARY

Build complete: ${TAG}
  commit  : ${have_commit}
  options : ${opts}

Configuration fingerprint (compare this with the other parties):
  ${fingerprint}

If any party's fingerprint differs, do not run: you would be executing different
protocols against each other, which hangs at best and is silently wrong at worst.
SUMMARY

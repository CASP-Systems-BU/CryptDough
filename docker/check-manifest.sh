#!/usr/bin/env bash
#
# Pre-flight check: does this party's local state match the agreed run manifest?
#
# Run this BEFORE launching. A mismatch in thread count or any compile-time option
# does not produce a useful error at run time -- it produces a hang (the port
# arithmetic diverges) or a silently wrong answer. Catching it here is far cheaper.

set -euo pipefail

MANIFEST=""
SRC_DIR="${SRC_DIR:-.}"
CERT_DIR=""

usage() {
    cat <<'USAGE'
Usage: check-manifest.sh --manifest FILE [--src DIR] [--certs DIR]

  --manifest FILE  The agreed run manifest (YAML).
  --src DIR        CryptDough checkout to verify the commit of (default: .).
  --certs DIR      Directory holding the peer certificates named in the manifest.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --src)      SRC_DIR="$2"; shift 2 ;;
        --certs)    CERT_DIR="$2"; shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

[[ -n "${MANIFEST}" ]] || { echo "ERROR: --manifest is required" >&2; usage; exit 1; }
[[ -f "${MANIFEST}" ]] || { echo "ERROR: no such manifest: ${MANIFEST}" >&2; exit 1; }

fail=0
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=1; }
note() { printf '  --    %s\n' "$*"; }

# Minimal YAML scalar reader: good enough for this flat manifest, and avoids making
# every party install a YAML library just to run a pre-flight check.
value_of() {
    sed -n "s/^[[:space:]]*$1:[[:space:]]*//p" "${MANIFEST}" \
        | head -1 | sed 's/[[:space:]]*#.*$//' | tr -d '"'"'"''
}

echo "Checking against ${MANIFEST}"
echo

# --- git commit -------------------------------------------------------------
want_commit="$(value_of git_commit)"
if [[ -n "${want_commit}" && "${want_commit}" != 0000000000000000000000000000000000000000 ]]; then
    have_commit="$(git -C "${SRC_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
    if [[ "${have_commit}" == "${want_commit}" ]]; then
        ok "git commit ${have_commit:0:12}"
    else
        bad "git commit is ${have_commit:0:12}, manifest wants ${want_commit:0:12}"
    fi
    if [[ -n "$(git -C "${SRC_DIR}" status --porcelain 2>/dev/null)" ]]; then
        bad "working tree has uncommitted changes; the other parties are building something else"
    else
        ok "working tree is clean"
    fi
else
    note "git_commit is a placeholder; skipping commit check"
fi

# --- compile-time options vs the CMake cache --------------------------------
CACHE="${SRC_DIR}/build/CMakeCache.txt"
if [[ -f "${CACHE}" ]]; then
    check_cache() {
        local key="$1" want="$2"
        local have
        have="$(sed -n "s/^${key}:[A-Z]*=//p" "${CACHE}" | head -1)"
        if [[ -z "${have}" ]]; then
            note "${key} not in CMake cache (default in effect)"
        elif [[ "${have}" == "${want}" ]]; then
            ok "${key}=${have}"
        else
            bad "${key}=${have}, manifest wants ${want}"
        fi
    }
    for key in PROTOCOL COMM COMM_THREADS TLS TRIPLES DEFAULT_BITWIDTH \
               SORT_PROTO BOOLEAN_ADDER DIVISION_CORRECTION; do
        want="$(value_of "${key}")"
        [[ -n "${want}" ]] && check_cache "${key}" "${want}"
    done
else
    note "no CMakeCache.txt under ${SRC_DIR}/build; build first, then re-run"
fi

# --- TLS material -----------------------------------------------------------
if [[ "$(value_of TLS)" == "ON" ]]; then
    if [[ -n "${CERT_DIR}" ]]; then
        while read -r cert; do
            [[ -z "${cert}" ]] && continue
            if [[ -f "${CERT_DIR}/${cert}" ]]; then
                fp="$(openssl x509 -in "${CERT_DIR}/${cert}" -noout -fingerprint -sha256 \
                      | sed 's/.*=//; s/://g' | tr 'A-Z' 'a-z')"
                ok "peer cert ${cert} present, SHA-256 ${fp:0:16}..."
            else
                bad "peer cert ${cert} missing from ${CERT_DIR}"
            fi
        done < <(sed -n 's/^[[:space:]]*cert:[[:space:]]*//p' "${MANIFEST}" | tr -d '"')
        echo
        note "Confirm each fingerprint with its owner over a channel an attacker"
        note "cannot also control. That out-of-band step is what makes pinning work."
    else
        note "--certs not given; skipping certificate checks"
    fi
fi

# --- the port range this party must open ------------------------------------
base_port="$(value_of base_port)"
threads="$(value_of threads)"
nparties="$(grep -c '^[[:space:]]*-[[:space:]]*rank:' "${MANIFEST}" || echo 0)"
if [[ -n "${base_port}" && -n "${threads}" && "${nparties}" -gt 0 ]]; then
    echo
    echo "Inbound ports each party must accept (num_parties=${nparties}, threads=${threads}):"
    for ((r = 0; r < nparties; r++)); do
        if (( r == 0 )); then
            printf '  rank 0: none (it only makes outbound connections)\n'
            continue
        fi
        printf '  rank %d: ' "$r"
        for ((j = 0; j < r; j++)); do
            start=$(( base_port + nparties * threads * j + threads * r ))
            printf '%d-%d (from rank %d) ' "${start}" "$(( start + threads - 1 ))" "$j"
        done
        printf '\n'
    done
fi

echo
if [[ "${fail}" -eq 0 ]]; then
    echo "Manifest check passed."
else
    echo "Manifest check FAILED -- do not launch until this is resolved." >&2
    exit 1
fi

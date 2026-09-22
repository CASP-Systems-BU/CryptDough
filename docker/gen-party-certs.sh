#!/usr/bin/env bash
#
# Generate this party's TLS identity for a cross-organizational CryptDough run.
#
# Each organization runs this ONCE, keeps the private key to itself, and sends the
# certificate (public, safe to email) to the other parties. Everyone then pins
# everyone else's certificate by SHA-256 fingerprint -- there is no CA, because with a
# small fixed set of mutually-distrusting parties there is no third party they would
# all agree to trust.
#
#   ./docker/gen-party-certs.sh --party 0 --out /scratch/adam/tls
#
# Then exchange <out>/party0.crt with the other parties OUT OF BAND and verify the
# printed fingerprint over a channel an attacker cannot also control.

set -euo pipefail

PARTY=""
OUT_DIR=""
DAYS=365

usage() {
    cat <<'USAGE'
Usage: gen-party-certs.sh --party N --out DIR [--days N]

  --party N   This party's rank (0, 1, 2). Used only to name the files.
  --out DIR   Where to write party<N>.key and party<N>.crt.
  --days N    Certificate lifetime in days (default: 365).
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --party) PARTY="$2"; shift 2 ;;
        --out)   OUT_DIR="$2"; shift 2 ;;
        --days)  DAYS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "${PARTY}" || -z "${OUT_DIR}" ]]; then
    echo "ERROR: --party and --out are both required" >&2
    usage
    exit 1
fi

mkdir -p "${OUT_DIR}"
chmod 700 "${OUT_DIR}"

KEY="${OUT_DIR}/party${PARTY}.key"
CRT="${OUT_DIR}/party${PARTY}.crt"

if [[ -e "${KEY}" || -e "${CRT}" ]]; then
    echo "ERROR: ${KEY} or ${CRT} already exists; refusing to overwrite." >&2
    echo "       Rotating a key means redistributing the certificate to every party." >&2
    exit 1
fi

# P-256 rather than RSA: faster handshakes, and there are H^2 x threads of them at
# startup (128 for 3 parties at 8 threads).
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
    -keyout "${KEY}" -out "${CRT}" \
    -days "${DAYS}" -nodes \
    -subj "/CN=cryptdough-party${PARTY}" \
    2>/dev/null

chmod 600 "${KEY}"
chmod 644 "${CRT}"

FINGERPRINT="$(openssl x509 -in "${CRT}" -noout -fingerprint -sha256 \
               | sed 's/.*=//; s/://g' | tr 'A-Z' 'a-z')"

cat <<SUMMARY

Party ${PARTY} identity written:
  private key : ${KEY}   (never leaves this machine, never goes in an image)
  certificate : ${CRT}   (send this to the other parties)

SHA-256 fingerprint:
  ${FINGERPRINT}

Next:
  1. Send ${CRT} to the other parties.
  2. Read the fingerprint above to them over a channel an attacker cannot control,
     and have them confirm it matches what they compute with:
       openssl x509 -in party${PARTY}.crt -noout -fingerprint -sha256
  3. Collect the other parties' certificates into this directory.

SUMMARY

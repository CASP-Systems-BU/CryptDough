#!/usr/bin/env bash
#
# CryptDough container entrypoint.
#
# Runs as root just long enough to install the mounted SSH key and start sshd,
# then drops to the unprivileged user before exec'ing the requested command
# (mpirun refuses to run as root, and we do not want experiment binaries running
# privileged either).
#
# Environment:
#   CDOUGH_SSHD=1        start sshd on port 2222 (needed for multi-node nocopy)
#   CDOUGH_USER          user to drop to             (default: cdough)
#   CDOUGH_SRC           source tree                 (default: /opt/cryptdough)
#   CDOUGH_KEY_DIR       mounted key directory       (default: /keys)
#                        expects id_ed25519 and/or authorized_keys
#   CDOUGH_HOSTNAME      override the container hostname (e.g. node0)

set -euo pipefail

CDOUGH_USER="${CDOUGH_USER:-cdough}"
CDOUGH_SRC="${CDOUGH_SRC:-/opt/cryptdough}"
CDOUGH_KEY_DIR="${CDOUGH_KEY_DIR:-/keys}"

log() { printf '[entrypoint] %s\n' "$*" >&2; }

user_home="$(getent passwd "${CDOUGH_USER}" | cut -d: -f6)"
if [[ -z "${user_home}" ]]; then
    log "FATAL: user ${CDOUGH_USER} does not exist in this image"
    exit 1
fi

# The build directory is a mount point in normal use, so it appears owned by root
# (or by the host uid) until we fix it. Without this, cmake cannot write its cache.
mkdir -p "${CDOUGH_SRC}/build"
if [[ "$(id -u)" -eq 0 ]]; then
    chown "${CDOUGH_USER}:${CDOUGH_USER}" "${CDOUGH_SRC}/build" 2>/dev/null || \
        log "warning: could not chown ${CDOUGH_SRC}/build (read-only or foreign uid?)"
fi

# ---------------------------------------------------------------------------
# secure-join compatibility symlink
#
# CMAKE_PREFIX_PATH covers every find_package/find_path/find_library call in
# CMakeLists.txt, but two lines bypass CMake's search entirely and hardcode a path
# into the source tree (only reached by PROTOCOL=2 with REAL triples):
#
#   CMakeLists.txt:154  link_directories(.../build/secure-join-install/lib)
#   CMakeLists.txt:167  target_include_directories(... .../build/secure-join-install/include/secureJoin/)
#
# /opt/cdough-deps has exactly that internal layout, so one symlink satisfies both.
# It must be made here rather than in the Dockerfile because build/ is normally a
# mounted volume, which would shadow anything baked into the image.
compat_link="${CDOUGH_SRC}/build/secure-join-install"
if [[ ! -e "${compat_link}" ]]; then
    ln -sfn /opt/cdough-deps "${compat_link}" 2>/dev/null \
        && log "linked build/secure-join-install -> /opt/cdough-deps" \
        || log "warning: could not create secure-join compatibility symlink"
fi

# ---------------------------------------------------------------------------
# SSH key material
# ---------------------------------------------------------------------------
install_keys() {
    local ssh_dir="${user_home}/.ssh"
    install -d -m 0700 -o "${CDOUGH_USER}" -g "${CDOUGH_USER}" "${ssh_dir}"

    if [[ -f "${CDOUGH_KEY_DIR}/id_ed25519" ]]; then
        # Copy rather than symlink: ssh rejects a key whose permissions are loose,
        # and a bind-mounted file carries the host's mode.
        install -m 0600 -o "${CDOUGH_USER}" -g "${CDOUGH_USER}" \
            "${CDOUGH_KEY_DIR}/id_ed25519" "${ssh_dir}/id_ed25519"
        log "installed private key"
    fi

    if [[ -f "${CDOUGH_KEY_DIR}/authorized_keys" ]]; then
        install -m 0600 -o "${CDOUGH_USER}" -g "${CDOUGH_USER}" \
            "${CDOUGH_KEY_DIR}/authorized_keys" "${ssh_dir}/authorized_keys"
        log "installed authorized_keys"
    elif [[ -f "${CDOUGH_KEY_DIR}/id_ed25519.pub" ]]; then
        install -m 0600 -o "${CDOUGH_USER}" -g "${CDOUGH_USER}" \
            "${CDOUGH_KEY_DIR}/id_ed25519.pub" "${ssh_dir}/authorized_keys"
        log "installed authorized_keys from id_ed25519.pub"
    fi
}

if [[ -d "${CDOUGH_KEY_DIR}" && "$(id -u)" -eq 0 ]]; then
    install_keys
fi

if [[ "${CDOUGH_SSHD:-0}" == "1" ]]; then
    if [[ "$(id -u)" -ne 0 ]]; then
        log "FATAL: CDOUGH_SSHD=1 requires the entrypoint to start as root"
        exit 1
    fi
    # Host keys are generated per container; the client config sets
    # StrictHostKeyChecking no, so churn is harmless here.
    ssh-keygen -A >/dev/null
    mkdir -p /run/sshd
    /usr/sbin/sshd
    log "sshd listening on port 2222"
fi

# ---------------------------------------------------------------------------
# Optional hostname override
#
# run_experiment.py and startmpc address peers as node0, node1, ... A container
# does not need to *be* named node0 to be reachable as node0 (that is what
# --add-host / --hostname on the docker run side is for), but scripts such as
# cluster-wan-sim.sh read the local hostname, so allow setting it.
# ---------------------------------------------------------------------------
if [[ -n "${CDOUGH_HOSTNAME:-}" && "$(id -u)" -eq 0 ]]; then
    hostname "${CDOUGH_HOSTNAME}" 2>/dev/null || \
        log "warning: could not set hostname (needs --privileged or --uts=host)"
fi

# ---------------------------------------------------------------------------
# Drop privileges and run
# ---------------------------------------------------------------------------
if [[ "$(id -u)" -eq 0 ]]; then
    # --login would reset the carefully constructed PATH/CMAKE_PREFIX_PATH, so use
    # runuser without it and let the image's ENV carry through.
    exec runuser -u "${CDOUGH_USER}" -- "$@"
fi

exec "$@"

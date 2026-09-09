# syntax=docker/dockerfile:1.7
#
# CryptDough experiment image.
#
# The third-party dependencies are
# installed to ${DEPS_PREFIX} (outside the source tree) and exposed to CMake via
# CMAKE_PREFIX_PATH, NOT to <source>/build/*-install where CMakeLists.txt hints for
# them. This means:
#   * a host working copy can be bind-mounted over /opt/cryptdough without hiding them
#   * /opt/cryptdough/build is disposable, so it can be a persistent named volume
# CMakeLists.txt needs no changes: every find_package/find_path/find_library call there
# uses HINTS, which are searched *in addition to* CMAKE_PREFIX_PATH.
#
# Build (from the repository root):
#   docker build -t cryptdough:latest -f docker/Dockerfile .
#
# Build it on hardware matching the machines that will run it: CMakeLists.txt adds
# -march=native on x86_64, and the dependencies compiled into this image inherit
# equivalent flags.

ARG UBUNTU_VERSION=24.04

##############################################################################
# Stage 1: base — toolchain, system libraries, Python, and the non-root user.
##############################################################################
FROM ubuntu:${UBUNTU_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Two groups of packages:
#  (a) everything scripts/setup/_setup_required.sh installs
#  (b) tools the run scripts shell out to that a minimal ubuntu image lacks.
#      Group (b) is the most common cause of a first-run failure, because these
#      happen to be preinstalled on a typical cloud VM.
RUN apt-get update && apt-get install -y --no-install-recommends \
        `# (a) from scripts/setup/_setup_required.sh` \
        git cmake pkg-config build-essential manpages-dev gfortran wget \
        libsqlite3-0 libsqlite3-dev \
        libsodium23 libsodium-dev \
        libopenmpi3 libopenmpi-dev openmpi-bin openmpi-common \
        libopenblas-dev libblas-dev \
        python3 python3-pip python3-venv \
        libtool autoconf automake \
        `# (b) required by startmpc / run_experiment.py / wan-sim.py` \
        openssh-client openssh-server \
        procps coreutils util-linux \
        dnsutils iproute2 iputils-ping \
        sudo ca-certificates perl \
        `# convenience for debugging inside a running container` \
        netcat-openbsd curl less nano \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 enforces PEP 668, so the plotting stack goes in a venv.
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
ENV PATH=/opt/venv/bin:$PATH

# mpirun refuses to run as root, and scripts/testing/run_multithreaded_test.sh
# exercises MPI as well as NOCOPY. uid 1001 avoids Ubuntu 24.04's stock
# "ubuntu" user at uid 1000.
ARG CDOUGH_USER=cdough
ARG CDOUGH_UID=1001
ARG CDOUGH_GID=1001
RUN groupadd -g ${CDOUGH_GID} ${CDOUGH_USER} \
    && useradd -m -u ${CDOUGH_UID} -g ${CDOUGH_GID} -s /bin/bash ${CDOUGH_USER} \
    && printf '%s ALL=(ALL) NOPASSWD: ALL\n' "${CDOUGH_USER}" > /etc/sudoers.d/90-${CDOUGH_USER} \
    && chmod 0440 /etc/sudoers.d/90-${CDOUGH_USER}

##############################################################################
# Stage 2: deps — the five source-built libraries.
#
# Each library gets its own RUN so that bumping one pin does not invalidate the
# others. libOTe is the long pole (20-40 min) and fetches boost/relic/sodium
# itself, so the build needs network access.
##############################################################################
FROM base AS deps

ARG DEPS_PREFIX=/opt/cdough-deps
ARG BUILD_JOBS=8

# Pins. libOTe/secure-join match scripts/setup/_clone_libote.sh and
# _setup_securejoin.sh. Blaze is pinned here to a tag because _setup_blaze.sh
# clones master with no checkout, which is not reproducible.
ARG NTL_VERSION=11.6.0
ARG LIBOTE_REPO=https://github.com/elimbaum/libOTe.git
ARG LIBOTE_COMMIT=587b325d2dfcab27a70f7954adc99a5cd65712d8
ARG SECUREJOIN_REPO=https://github.com/CASP-Systems-BU/secure-join.git
ARG SECUREJOIN_COMMIT=98fda2e6780738926bc53c90e801c449e3ba89e8
ARG BLAZE_REPO=https://bitbucket.org/blaze-lib/blaze.git
ARG BLAZE_TAG=v3.8.2

WORKDIR /tmp/deps
RUN mkdir -p ${DEPS_PREFIX}

# --- NTL (no GMP, matching scripts/setup/_setup_ntl.sh) ---------------------
RUN echo "==== Building NTL ${NTL_VERSION}..." \
    && wget -q https://libntl.org/ntl-${NTL_VERSION}.tar.gz \
    && tar -xzf ntl-${NTL_VERSION}.tar.gz \
    && rm ntl-${NTL_VERSION}.tar.gz \
    && cd ntl-${NTL_VERSION}/src \
    && ./configure PREFIX=${DEPS_PREFIX} NTL_GMP_LIP=off \
    && make -j${BUILD_JOBS} \
    && make install \
    && cd /tmp/deps && rm -rf ntl-${NTL_VERSION} \
    && echo "==== NTL installed"

# --- clone libOTe once; cryptoTools lives inside it -------------------------
RUN echo "==== Cloning libOTe @ ${LIBOTE_COMMIT}..." \
    && git clone ${LIBOTE_REPO} libOTe \
    && cd libOTe \
    && git checkout ${LIBOTE_COMMIT} \
    && git submodule update --init

# --- cryptoTools (always needed; see CMakeLists.txt:157) --------------------
#
# The thirdparty flags here must MATCH the libOTe step below. cryptoTools vendors
# coproto and installs coproto/config.h into the shared prefix; building it bare
# (as scripts/setup/_setup_cryptotools.sh does) writes a header with
# COPROTO_ENABLE_BOOST undefined, and the later libOTe build -- which does enable
# boost -- does not overwrite it. The stale header compiles cp::asioConnect out of
# existence, which breaks tests/test_libote.cpp under PROTOCOL=2 with REAL triples.
RUN echo "==== Building cryptoTools..." \
    && cd libOTe/cryptoTools \
    && python3 build.py --par=${BUILD_JOBS} --boost --sodium --relic \
        -DSODIUM_MONTGOMERY=false --install=${DEPS_PREFIX} \
    && echo "==== cryptoTools installed"

# --- libOTe (only linked for PROTOCOL=2 with REAL triples) ------------------
RUN echo "==== Building libOTe..." \
    && cd libOTe \
    && if [ "$(uname -m)" = "x86_64" ]; then \
           SSE_OPTION="-DENABLE_SSE=ON"; \
       else \
           SSE_OPTION="-DENABLE_SSE=OFF -DENABLE_AVX=OFF"; \
       fi \
    && python3 build.py --par=${BUILD_JOBS} --boost --sodium --relic \
        -DSODIUM_MONTGOMERY=false --all --install=${DEPS_PREFIX} ${SSE_OPTION} \
    && echo "==== libOTe installed"

# --- secure-join (2PC OPRF) ------------------------------------------------
RUN echo "==== Building SecureJoin..." \
    && git clone ${SECUREJOIN_REPO} secure-join \
    && cd secure-join \
    && git checkout ${SECUREJOIN_COMMIT} \
    && if [ "$(uname -m)" = "x86_64" ]; then \
           SSE_OPTION="-DENABLE_SSE=ON"; \
       else \
           SSE_OPTION="-DENABLE_SSE=OFF -DENABLE_AVX=OFF -DSECUREJOIN_ENABLE_SSE=OFF"; \
       fi \
    && python3 build.py -DSODIUM_MONTGOMERY=false --install=${DEPS_PREFIX} ${SSE_OPTION} \
    && echo "==== SecureJoin installed"

# --- Blaze (header-only; USE_BLAZE defaults ON in CMakeLists.txt:173) -------
RUN echo "==== Building Blaze ${BLAZE_TAG}..." \
    && git clone --depth 1 --branch ${BLAZE_TAG} ${BLAZE_REPO} blaze \
    && cd blaze \
    && cmake -DCMAKE_INSTALL_PREFIX=${DEPS_PREFIX} . \
    && make install \
    && echo "==== Blaze installed"

##############################################################################
# Stage 3: runtime — deps + source + launcher wiring.
##############################################################################
FROM base AS runtime

ARG DEPS_PREFIX=/opt/cdough-deps
ARG CDOUGH_USER=cdough
ARG CDOUGH_SRC=/opt/cryptdough

COPY --from=deps ${DEPS_PREFIX} ${DEPS_PREFIX}

# The keystone: CMakeLists.txt's HINTS point into <source>/build, but every one of
# those calls also searches CMAKE_PREFIX_PATH, so the deps are found here instead
# and survive a bind-mount of the source tree.
ENV CMAKE_PREFIX_PATH=${DEPS_PREFIX} \
    LD_LIBRARY_PATH=${DEPS_PREFIX}/lib:${DEPS_PREFIX}/lib64 \
    CDOUGH_SRC=${CDOUGH_SRC}

COPY --chown=${CDOUGH_USER}:${CDOUGH_USER} . ${CDOUGH_SRC}

# Replaces the ~/bin symlink and .bashrc PATH edit in _setup_required.sh:12-26.
# The target is inside the source tree, so the symlink keeps resolving when a host
# working copy is mounted at the same path.
RUN ln -sf ${CDOUGH_SRC}/include/backend/nocopy_communicator/startmpc/startmpc \
        /usr/local/bin/startmpc \
    && chmod +x ${CDOUGH_SRC}/include/backend/nocopy_communicator/startmpc/startmpc \
    && mkdir -p ${CDOUGH_SRC}/build \
    && chown ${CDOUGH_USER}:${CDOUGH_USER} ${CDOUGH_SRC}/build

# SSH on 2222. Under --network host the container shares the host's network
# namespace, so port 22 belongs to the host's own sshd. The client-side "Host node*"
# block is what lets startmpc and run_experiment.py call plain `ssh`/`scp` with no
# port flag and still reach a peer container.
COPY docker/sshd_config /etc/ssh/sshd_config.d/cdough.conf
COPY docker/ssh_config  /etc/ssh/ssh_config.d/cdough.conf
RUN mkdir -p /run/sshd \
    && chmod 0644 /etc/ssh/sshd_config.d/cdough.conf /etc/ssh/ssh_config.d/cdough.conf

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# run_experiment.py resolves its build directory as <cwd>/../build, so it must be
# invoked from the build directory — same as the bare-metal workflow.
WORKDIR ${CDOUGH_SRC}/build

EXPOSE 2222

# The entrypoint starts as root (to launch sshd) and drops to ${CDOUGH_USER}.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

# Running CryptDough in Docker

This directory packages CryptDough so it can be dropped onto a machine and run
experiments. It targets
**one container per physical machine**, using the **nocopy** communicator.

- [What the image gives you](#what-the-image-gives-you)
- [Quick start](#quick-start)
- [Full dependency list](#full-dependency-list)
- [How the communicator works in containers](#how-the-communicator-works-in-containers)
- [Iterating on source](#iterating-on-source)
- [Troubleshooting](#troubleshooting)

## What the image gives you

`docker build` produces an image where every dependency is already compiled, so
`run_experiment.py` works exactly as it does on a hand-provisioned node:

```bash
docker exec -it -u cdough cdough ../scripts/run_experiment.py -s lan -c nocopy -n 4 -T 8 micro_sorting
```

**What it does not do.** A container is not a VM — it shares the host kernel and runs
processes natively, which is why it is acceptable for benchmarking. Three consequences:

| | Why | What to do |
|---|---|---|
| No CPU abstraction | `CMakeLists.txt:26-30` adds `-march=native` on x86_64, and the dependencies baked into the image inherit equivalent flags | Build the image on hardware matching the cluster. A binary built on a newer CPU can `SIGILL` on an older one. |
| No kernel control | `/sys` is read-only in a container, so `_setup_required.sh:17`'s transparent-hugepage write cannot happen inside | Set it once on each **host** (see below) |
| Networking is not free | The default bridge network NATs and adds latency, which would corrupt communication measurements | Always use `--network host` for real runs (`run-node.sh` does) |

Experiment binaries are compiled **at run time inside the container**, on the machine
that will run them. Only the third-party dependencies are prebuilt into the image.

## Quick start

### 1. Host preparation (once per machine)

```bash
# Docker Engine, and permission to talk to it without sudo
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"      # log out and back in

# Transparent hugepages: host-only, /sys is read-only inside a container
sudo sh -c 'echo always > /sys/kernel/mm/transparent_hugepage/enabled'
```

### 2. Build the image

From the repository root, on a machine matching the cluster's CPU:

```bash
docker build -t cryptdough:latest -f docker/Dockerfile .
```

Expect 30-60 minutes; libOTe dominates and fetches boost/relic/sodium during the
build, so network access is required. Tune parallelism with
`--build-arg BUILD_JOBS=32`.

### 3. Distribute it

```bash
# With a registry
docker tag cryptdough:latest ghcr.io/<you>/cryptdough:latest
docker push ghcr.io/<you>/cryptdough:latest        # then docker pull on each node

# Without one
docker save cryptdough:latest | gzip > cryptdough.tar.gz
scp cryptdough.tar.gz peer: && ssh peer 'gunzip -c cryptdough.tar.gz | docker load'
```

### 4. Create the cluster keypair (once, shared by all nodes)

```bash
mkdir -p ~/.cdough-cluster
ssh-keygen -t ed25519 -N '' -f ~/.cdough-cluster/id_ed25519
cp ~/.cdough-cluster/id_ed25519.pub ~/.cdough-cluster/authorized_keys
# copy the whole directory to every machine
```

### 5. Start a container on each machine

Run this on **every** machine, with the **same** `--nodes` list. The list is
positional: the first entry becomes `node0`, the second `node1`, and so on.

```bash
./docker/run-node.sh --nodes blinky,pinky,inky,clyde
```

Real machines are rarely named `node0..node3`, but `run_experiment.py` assumes that
numbering. Rather than needing root to edit `/etc/hosts` on the host, `run-node.sh`
resolves each machine and injects `node0..nodeN-1` aliases into the container's own
`/etc/hosts` with `--add-host`.

### 6. Run experiments from the node0 machine

```bash
docker exec -it -u cdough cdough ../scripts/run_experiment.py -s lan -c nocopy -n 4 -T 8 micro_sorting
```

Results land in the `cdough-build` named volume, which survives `docker rm`:
`output.json` and `.experiment.json` in `/opt/cryptdough/build`.

## Full dependency list

Everything here is installed by the Dockerfile; none of it needs to exist on the
target machine.

### apt — from `scripts/setup/_setup_required.sh`

```
git cmake pkg-config build-essential manpages-dev gfortran wget
libsqlite3-0 libsqlite3-dev  libsodium23 libsodium-dev
libopenmpi3 libopenmpi-dev openmpi-bin openmpi-common
libopenblas-dev libblas-dev  python3 python3-pip
libtool autoconf automake
```

OpenMPI is **not optional even for nocopy**: `CMakeLists.txt:121` calls
`find_package(MPI REQUIRED)` outside any communicator guard.

### apt — additionally required, and absent from a minimal `ubuntu:24.04`

These are preinstalled by accident on a typical cloud VM, which is why they are
missing from `_setup_required.sh`. They are the most likely cause of a first-run
failure on a lean base image.

| Package | Provides | Needed by |
|---|---|---|
| `openssh-client` | `ssh`, `scp` | `startmpc` remote mode; `run_experiment.py:551` |
| `openssh-server` | `sshd` | receiving parties (node1...) in remote mode |
| `procps` | `pkill` | `startmpc` cleanup trap (lines 16, 22) |
| `coreutils` | `stdbuf` | `startmpc` lines 54, 87 (line-buffered output) |
| `util-linux` | `runuser` | entrypoint privilege drop |
| `dnsutils` | `dig` | `run_experiment.py:342`; `wan-sim.py:17` |
| `iproute2` | `ip`, `tc` | interface/subnet discovery; WAN simulation |
| `iputils-ping` | `ping` | `run_experiment.py:891` latency checks |
| `sudo` | `sudo` | `wan-sim.py` runs `sudo tc qdisc` |
| `ca-certificates` | TLS roots | cloning at build time |
| `perl` | `perl` | NTL's `./configure` is a Perl script |
| `python3-venv` | venv | Ubuntu 24.04 enforces PEP 668 |

### Python — `requirements.txt`, installed to `/opt/venv`

`numpy`, `matplotlib`, `pandas`, `sphinx_rtd_theme`, `breathe`, `myst_parser`

### Source-built libraries, installed to `/opt/cdough-deps`

| Library | Source | Pin |
|---|---|---|
| NTL 11.6.0 | `libntl.org` tarball | version; built `NTL_GMP_LIP=off` |
| cryptoTools | `github.com/elimbaum/libOTe`, `cryptoTools/` | `587b325d2dfcab27a70f7954adc99a5cd65712d8` |
| libOTe | same fork | same commit; `--boost --sodium --relic --all` |
| secure-join | `github.com/CASP-Systems-BU/secure-join` | `98fda2e6780738926bc53c90e801c449e3ba89e8` |
| Blaze | `bitbucket.org/blaze-lib/blaze` | `v3.8.2` — pinned here; `_setup_blaze.sh` clones master with no checkout |

**Why `/opt/cdough-deps` and not `<source>/build/*-install`?** `CMakeLists.txt:133-185`
hints for the dependencies inside the source tree. If they lived there, bind-mounting a
host working copy over `/opt/cryptdough` would hide them. Every one of those calls is a
`find_package`/`find_path`/`find_library` with `HINTS`, all of which also search
`CMAKE_PREFIX_PATH` — so the image sets `CMAKE_PREFIX_PATH=/opt/cdough-deps` and
`CMakeLists.txt` needs no change. The payoff: `build/` becomes disposable, so it can be
a named volume that persists the CMake cache and compiled binaries across runs.

## How the communicator works in containers

`startmpc` is a bash script that does two things: pick a base port, and start N copies
of the binary with five environment variables set (`STARTMPC_EXEC_MODE`, `_BASE_PORT`,
`_HOST_COUNT`, `_HOST_RANK`, `_HOST_LIST`). The C++ side reads only those variables
(`startmpc.h:77-113`).

- **Local mode** (`startmpc -n 3 ./prog`, i.e. `-s same`) forks N children that talk
  over `127.0.0.1`. Works in a single container with no extra configuration.
- **Remote mode** (`startmpc -n 3 -h node0,node1,node2`) runs each party over `ssh`
  and `scp`s the binary. This is the multi-machine path.

### The port-2222 arrangement

Under `--network host` the container shares the host's network namespace, so port 22
already belongs to the host's own sshd. The image therefore runs its sshd on **2222**
and ships an `/etc/ssh/ssh_config.d/cdough.conf` with a `Host node*` block setting
`Port 2222`. `startmpc` and `run_experiment.py` call plain `ssh`/`scp` with no port
flag, so this client-side block is what makes them work **without modifying any file in
the repository**.

If your cluster uses a different node prefix (`run_experiment.py -x`), add it to the
`Host` patterns in [`ssh_config`](ssh_config).

### Ports to open between cluster machines

- **TCP 2222** — inter-container SSH.
- **TCP 10000-51000** — `startmpc` picks a random base port in `[10000, 50000]`
  (line 119) and uses up to `host_count^2 * threads` consecutive ports above it
  (`startmpc.h:126-137`). Four parties at 8 threads is 128 ports.

### Fallback: no SSH at all

Because the C++ side reads only environment variables, you can skip `startmpc`
entirely. [`run-party.sh`](run-party.sh) launches one party per machine with the five
variables set directly — no sshd, no keys, and a deterministic base port so a narrow
firewall range suffices. You give up `run_experiment.py`'s build, scp, repetition and
JSON aggregation.

## Iterating on source

Bind-mount your working copy; the dependencies live outside the source tree, so they
survive:

```bash
./docker/run-node.sh --nodes blinky,pinky,inky,clyde --src ~/CryptDough
```

## Troubleshooting

**`find_package` cannot find cryptoTools / Blaze / NTL / MPI.**
`CMAKE_PREFIX_PATH` is not reaching CMake. Check `docker exec -u cdough cdough env | grep CMAKE`.
As a fallback, symlink the prefixes into the build directory under the names
`CMakeLists.txt` hints for:
```bash
docker exec -u cdough cdough bash -c 'cd /opt/cryptdough/build &&
  ln -sfn /opt/cdough-deps ntl-install &&
  ln -sfn /opt/cdough-deps libOTe-install &&
  ln -sfn /opt/cdough-deps secure-join-install &&
  ln -sfn /opt/cdough-deps blaze-install'
```

**`ssh node1` hangs or is refused.** Check the peer's container is up
(`docker ps`), that 2222 is reachable (`nc -vz node1 2222`), and that both machines
were started with the same `--nodes` list. Confirm the alias resolves:
`docker exec -u cdough cdough getent hosts node1`.

**`mpirun` refuses to run as root.** You almost certainly omitted `-u cdough`.
`docker exec` bypasses the ENTRYPOINT that normally drops privileges, so
`docker exec cdough whoami` prints `root` while `docker exec -u cdough cdough whoami`
prints `cdough`. This affects MPI only; nocopy runs either way, but as root it leaves
root-owned files in the build volume.

**A run crashes in a mysterious way across nodes.** The main README warns that mismatched
binaries on different hosts cause exactly this. `run_experiment.py` scp's the binary
each run, but if you built by hand, rebuild everywhere.

**Illegal instruction (SIGILL).** The image was built on a CPU newer than this host.
Rebuild it here, or rebuild the dependencies with an explicit baseline such as
`-march=x86-64-v3`.

**You changed a dependency pin and the image did not pick it up.** The legacy builder
(what Ubuntu's `docker.io` gives you, since it ships without buildx) does not invalidate
`COPY --from=<stage>` when the source stage is rebuilt. It will happily rebuild the whole
`deps` stage and then serve the final stage's `COPY --from=deps` from cache, so the new
libraries never reach the image. Symptom: the build log shows your dependency rebuilding,
but the installed file's mtime is from an older build:

```bash
docker run --rm cryptdough:latest stat -c '%y %n' /opt/cdough-deps/include/coproto/config.h
```

Rebuild with `--no-cache` after any dependency change, or install `docker-buildx`, which
tracks inter-stage dependencies correctly.

**Ports still in use after a failed run.** `startmpc` randomizes its base port for this
reason, and `run_experiment.py` sleeps 30s between nocopy repetitions. If a run wedges,
`docker exec -u cdough cdough pkill -f <binary>` on every machine.

**WAN simulation does nothing.** `wan-sim.py` runs `sudo tc qdisc`, which needs
`--cap-add NET_ADMIN`. Restart with `run-node.sh --wan`.

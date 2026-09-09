# Task 0007 — Dockerize CryptDough

## Metadata
- Task ID: 0007
- Title: Dockerize CryptDough for portable experiment execution
- Requested by: Adam Godel
- Owner: Adam Godel
- Date: 2026-09-09
- Status: In Progress (validated on the blinky/pinky/inky/clyde cluster)
- Estimated effort: 1-2 days (image build alone is 30-60 min per iteration)
- Target completion date: TBD
- Related issue or PR: n/a
- Related documents: [README](../README.md), [sosp-replication/README](../sosp-replication/README.md)
- Branch: logistic-regression

## Problem Statement
- **What is the request?** Package CryptDough as a Docker image so the library can be sent
  to an unspecified machine and run experiments with minimal setup, with a complete
  dependency list captured inside the container.
- **Why does this matter now?** `scripts/setup/setup.sh` assumes Ubuntu with `apt` and
  `sudo`, then clones and compiles five third-party projects from source (NTL,
  cryptoTools, libOTe, secure-join, Blaze). That is a 30-60 minute process with many
  failure modes: a missing apt package, a non-Ubuntu distro, a pre-C++20 compiler, or an
  unpinned Bitbucket clone that has drifted. Setup also mutates the host outside the repo
  (`~/bin`, `~/.bashrc`, a kernel sysfs knob). Every new machine repeats this cost.

## Goals
- Goal 1: A single `docker build` produces an image where
  `../scripts/run_experiment.py -s lan -c nocopy ... <experiment>` works unmodified.
- Goal 2: The full dependency set (apt, Python, source-built libraries with pinned
  commits) is declared in one reviewable file.
- Goal 3: The nocopy communicator works both within a single container and across one
  container per physical machine in a real cluster.
- Goal 4: Benchmark numbers inside the container match bare metal within noise.

## Non-Goals
- Non-goal 1: Containerizing the SOSP baselines (TVA, ORQ, Pigeon, Piranha, MP-SPDZ).
- Non-goal 2: Supporting the MPI communicator across containers. MPI must still be
  installed (CMake requires it) and must work locally, but cross-node MPI is out of scope.
- Non-goal 3: Kubernetes, Swarm, or any orchestrator beyond a compose file used as a
  single-machine smoke test.
- Non-goal 4: Changing `CMakeLists.txt` or any existing script.

## Scope
- **In scope:** a new `docker/` directory (Dockerfile, entrypoint, ssh config, run
  wrappers, compose file, operator README); this task document; the `tasks/tasks.md` line.
- **Out of scope:** the existing build system, the setup scripts, the experiment sources.

## Impact Assessment
- **User impact:** additive. The bare-metal `setup.sh` path keeps working untouched.
- **Performance impact:** intended to be zero. Containers run natively on the host kernel;
  `--network host` removes NAT and the veth pair. Two caveats are handled explicitly:
  transparent hugepages must be set on the host, and `/dev/shm` must be enlarged for the
  local MPI tests.
- **Security and privacy impact:** the image runs an `sshd` on port 2222 so that
  `startmpc` remote mode can reach peer containers. This mirrors the SSH access the
  bare-metal cluster setup already requires, which the main README already flags as a
  benchmarking-only concession that breaks the non-collusion assumption. An SSH-free
  fallback is provided.
- **Backward compatibility impact:** none; no existing file is modified.

## Context

### Relevant files and modules
- `CMakeLists.txt` — dependency discovery and compiler flags.
- `scripts/setup/*.sh` — the dependency recipe being translated into the Dockerfile.
- `scripts/run_experiment.py` — the execution harness that must keep working verbatim.
- `include/backend/nocopy_communicator/startmpc/{startmpc,startmpc.h}` — the nocopy
  launcher and its connection setup.
- `include/backend/nocopy_communicator/network_utils.h` — socket bind/connect.
- `scripts/profiling/comm/{cluster-wan-sim.sh,wan-sim.py}` — WAN simulation.

### Constraints discovered while reading the build system
1. **MPI is unconditionally required.** `CMakeLists.txt:121` calls
   `find_package(MPI REQUIRED)` outside any `if(COMM ...)` guard. The OpenMPI dev packages
   must be in the image even though we only use nocopy.
2. **Dependency hints point into the source tree.** `CMakeLists.txt:133-185` hints NTL,
   libOTe, secureJoin, cryptoTools and Blaze at `${CMAKE_CURRENT_SOURCE_DIR}/build/*-install`.
   Installing deps there would make them vanish the moment a host working copy is
   bind-mounted over the image's source. All those calls are `find_package`/`find_path`/
   `find_library` with `HINTS`, which also search `CMAKE_PREFIX_PATH`.
3. **`-march=native` on x86_64** (`CMakeLists.txt:26-30`). Experiment binaries must be
   compiled on the machine that runs them; the prebuilt deps carry the same hazard, so the
   image should be built on the target hardware generation.
4. **`startmpc` is a bash launcher.** It picks a random base port in `[10000, 50000]` and
   starts N copies of the binary with five env vars set. Local mode forks children on this
   host; remote mode (`-h node0,node1,...`) runs each party over `ssh` (line 87) and
   optionally `scp`s the binary (line 76). Cleanup uses `pkill`.
5. **The C++ side reads only env vars** (`startmpc.h:77-113`): `STARTMPC_EXEC_MODE`,
   `STARTMPC_BASE_PORT`, `STARTMPC_HOST_COUNT`, `STARTMPC_HOST_RANK`, `STARTMPC_HOST_LIST`.
   Party *i* connects to all *j > i* and listens for all *j < i* on
   `base_port + host_count*threads*i + threads*j`, so the span is
   `base_port .. base_port + host_count^2 * threads`. SSH is therefore a convenience, not
   a requirement.
6. **Peer hostnames resolve via `getaddrinfo`** (`network_utils.h:66-75`) and
   `socket_create` binds `INADDR_ANY`. With `--network host`, `node0`/`node1` resolve from
   the host's `/etc/hosts`, which `_update_hostfile.sh` already populates.
7. **`run_experiment.py -s lan` needs cluster tooling:** `dig` and `ip route` (lines
   334-351), `scp` of the binary to the *same absolute path* on peers (line 551), and
   `ping` (line 891).
8. **WAN simulation needs privileges:** `wan-sim.py:51-55` runs `sudo tc qdisc`, requiring
   `iproute2`, passwordless `sudo`, and `--cap-add NET_ADMIN`.
9. **`mpirun` refuses to run as root**, and `run_multithreaded_test.sh` exercises both
   communicators, so the image needs a non-root user.
10. **No external data files.** Query suites generate data in-process into an in-memory
    SQLite DB (`bench/queries/tpch/q1.cpp:66`), so no data volumes are needed.

### Dependencies
See the "Full dependency list" in `docker/README.md`. Summary: the apt set from
`_setup_required.sh`; an additional apt set the run scripts need but that is absent from a
minimal `ubuntu:24.04` (`openssh-client`, `openssh-server`, `procps`, `dnsutils`,
`iproute2`, `iputils-ping`, `sudo`, `ca-certificates`); `requirements.txt` in a venv (PEP
668); and five source-built libraries at pinned commits.

## Alternatives

### Option A: Deps inside the source tree at `build/*-install`
Mirrors what `setup.sh` does today, so `CMakeLists.txt` finds everything with no
environment variables.

Pros:
- Zero divergence from the documented bare-metal layout.
- No reliance on `CMAKE_PREFIX_PATH` semantics.

Cons:
- Bind-mounting a host working copy over `/opt/cryptdough` hides the deps, breaking the
  iteration workflow entirely.
- `build/` can no longer be a persistent volume, so every container restart recompiles
  from scratch.

### Option B: Deps at `/opt/cdough-deps` exposed via `CMAKE_PREFIX_PATH`
Install outside the source tree; set `ENV CMAKE_PREFIX_PATH=/opt/cdough-deps`.

Pros:
- Source can be bind-mounted freely; deps survive.
- `build/` becomes disposable, so it can be a named volume that persists the CMake cache
  and compiled binaries across runs.
- Requires no change to `CMakeLists.txt`.

Cons:
- Depends on `find_package`/`find_path` honoring `CMAKE_PREFIX_PATH` in addition to their
  `HINTS`. Must be verified early; the fallback is to symlink the install prefixes into
  `build/` under the hinted names.

### Option C (communicator): SSH-free, env-var launch only
Set the five `STARTMPC_*` variables per container and skip `startmpc`.

Pros:
- No sshd, no keys, no extra attack surface.
- Deterministic base port, so a narrow firewall range suffices.

Cons:
- Loses `run_experiment.py`'s build, `scp`, repetition and JSON aggregation orchestration.
- Parties must be started manually within the connection timeout.

### Recommendation
- **Recommended: Option B for dependency layout, plus SSH-on-port-2222 for the
  communicator, with Option C shipped as a documented fallback.**
- Option B is the only layout that supports both a self-contained image and a
  bind-mounted working copy. The SSH approach keeps `run_experiment.py` working verbatim,
  which is the whole point of the exercise; port 2222 avoids colliding with the host's own
  sshd under `--network host`, and a shipped `Host node*` client config block means no
  repo file has to learn about the port.
- Open questions needing confirmation: none outstanding. Topology (one container per
  physical machine), build timing (deps prebuilt, experiments at run time), dependency
  scope (everything), and source handling (baked in with bind-mount override) were all
  confirmed before drafting.

## Implementation Plan
1. `docker/.dockerignore` — keep `build/`, `.git/`, `sosp-replication/data/` out of the
   build context.
2. `docker/Dockerfile` — three stages: `base` (apt + venv + non-root `cdough` user),
   `deps` (five libraries into `/opt/cdough-deps`, one `RUN` each for layer caching),
   final (copy deps, copy source, symlink `startmpc`, sshd config, `CMAKE_PREFIX_PATH`).
3. `docker/entrypoint.sh` — install mounted SSH key, optionally start `sshd`, drop to
   `cdough`, `exec "$@"`.
4. `docker/ssh_config`, `docker/sshd_config` — the `Host node*` / port 2222 pair.
5. `docker/run-node.sh` — canonical `docker run` for a cluster node.
6. `docker/run-party.sh` — the SSH-free fallback launcher.
7. `docker/compose.yaml` — single-machine multi-container smoke test.
8. `docker/README.md` — operator guide and the full dependency list.
9. Update `tasks/tasks.md`.

## Risks and Mitigations
- **Risk:** `-march=native` in the prebuilt deps causes `SIGILL` when the image is built on
  a newer CPU than the target.
  - **Mitigation:** build the image on the cluster hardware; `docker build` is one command.
    For mixed fleets, rebuild deps with an explicit `-march=x86-64-v3` baseline.
- **Risk:** Blaze is cloned from master with no `git checkout` in `_setup_blaze.sh`, so the
  image is not reproducible across rebuild dates.
  - **Mitigation:** pin a tag in the Dockerfile.
- **Risk:** libOTe's `build.py` fetches boost/relic/sodium during `docker build`; a network
  blip fails a 30-minute build.
  - **Mitigation:** isolate it in its own stage and `RUN` layer so retries are cheap; push
    the finished image to a registry so it is built once.
- **Risk:** `CMAKE_PREFIX_PATH` is not honored and no dependency is found.
  - **Mitigation:** verified as verification step 1; fallback is symlinking the prefixes
    into `build/` under the hinted names.
- **Risk:** container overhead silently taxes the benchmark, invalidating results.
  - **Mitigation:** verification step 6 compares container against bare metal on the same
    configuration.

## Rollback and Recovery
- **Rollback plan:** delete the `docker/` directory. Nothing outside it is modified, so
  the bare-metal path is unaffected.
- **Recovery steps:** on a broken image, `docker rmi` and rebuild; the named `build/`
  volume can be removed with `docker volume rm cdough-build` to force a clean recompile.
- **Monitoring after release:** confirm `build/output.json` carries plausible timings on
  the first cluster run, and diff against the most recent bare-metal numbers.

## Validation Plan
- **Automated checks:** `run_multithreaded_test.sh` for protocols 3, 4 and 2 inside the
  container; protocol 2 is the only path that exercises the libOTe/secure-join link.
- **Manual verification:** CMake configure finds every dependency; single-container nocopy
  and MPI smoke tests; multi-container SSH path via compose; a real cluster run of
  `micro_sorting` with `-s lan -c nocopy`; a bare-metal comparison run.
- **Expected success criteria:** all tests pass under both communicators; a cluster
  experiment produces `build/output.json` with non-zero timings; container and bare-metal
  timings agree within run-to-run noise.

## Definition of Done
- `docker build` succeeds from a clean checkout on a Linux x86_64 host.
- `run_experiment.py` runs unmodified inside the container for `-s same` and `-s lan`.
- The full test suite passes for protocols 2, 3 and 4.
- Tests updated or added: none required; the existing suite is the acceptance test.
- Documentation updated: `docker/README.md` written; main `README.md` gains a pointer to it.

## Verification Results (2026-09-09, blinky/pinky/inky/clyde)

Cluster: 4x Ubuntu 24.04, x86_64, 192 cores, ~251 GB RAM, private 192.168.100.0/24
LAN at ~0.1 ms RTT. Docker 29.1.3 on all four.

### Passed
- Image builds (1.65 GB, ~5 min at `-j48`); loads on peers in ~14 s each.
- CMake configure finds every dependency from `/opt/cdough-deps` via
  `CMAKE_PREFIX_PATH` with no `build/*-install` in the source tree.
- Single container: `test_primitives` passes under nocopy and under MPI.
- Full suite protocol 3, both communicators: `[[ All 3PC 1 thread MPI tests passed! ]]`,
  exit 0.
- Three-machine LAN run: `run_experiment.py -s lan -c nocopy` works unmodified; the
  binary is scp'd into the peer containers and all tests pass.
- SSH proven to terminate *inside* the peer container (`/opt/cryptdough` and
  `/opt/cdough-deps` exist there; they do not exist on the peer host).
- `--add-host` works under `--network host`, so `node0..node3` map to real machine
  names without editing `/etc/hosts` or needing root.
- SSH-free fallback (`run-party.sh`) works across three machines.
- `micro_sorting`, 3PC, 2^20 rows, 8 worker + 4 comm threads, containers on three
  separate machines over the LAN: Quicksort 3.18 s, Bitonic 7.04 s, Pairwise 6.71 s,
  Radix 6.50 s, Overall 23.43 s.
- Container vs bare metal on an idle machine, stable primitives (AND, MULT, EQ, PPA):
  parity within noise. High-variance ops (EQ_A, GR_A, RCA<) swing equally in both, so
  the variance is loopback-nocopy noise, not container overhead. Caveat: the container
  has gcc 13.3 and the bare-metal build gcc 12.4, so this is not a pure isolation of
  container cost.

### Bugs found and fixed during validation
1. `docker exec` bypasses the ENTRYPOINT that drops privileges, so it runs as root.
   All docs and `run-node.sh` hints now use `-u cdough`.
2. Ubuntu's `docker.io` ships without buildx, so the BuildKit-only
   `docker/Dockerfile.dockerignore` was silently ignored. Moved to a root
   `.dockerignore`, which both builders honor.
3. `CMakeLists.txt:154,167` hardcode `build/secure-join-install` into the source tree,
   bypassing CMake's search. The entrypoint now creates that symlink at container
   start (it cannot be baked into the image because `build/` is a volume).
4. Under `--network host`, port 2222 is a host-wide resource, so only one CryptDough
   container can run per machine. A second one starts but its sshd cannot bind, and
   the failure surfaces much later as a confusing "No such file or directory" from
   `stdbuf`. `run-node.sh` now detects this and refuses with a clear message.

### Open issue
PROTOCOL=2 with REAL Beaver triples does not compile in the container. Only this
configuration is affected; the test suite uses DUMMY triples and passes.

### Operational note
The legacy (non-BuildKit) builder does not invalidate `COPY --from=<stage>` when the
source stage is rebuilt. Changing a dependency pin requires `--no-cache`, or install
`docker-buildx`. This cost two misleading rebuild cycles during validation and is
documented in `docker/README.md`.

## Approval
- [x] Task document reviewed
- [x] Approved to implement
- Approver: Adam Godel
- Approval date: 2026-09-09

## Change Log
- 2026-09-09: Initial draft created and approved.
- 2026-09-09: Validated end-to-end on the blinky/pinky/inky/clyde cluster; four bugs found and fixed; one open issue (2PC with REAL triples) documented.

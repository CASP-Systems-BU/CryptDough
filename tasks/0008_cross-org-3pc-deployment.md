# Task 0008 — Real 3PC across three organizations

## Metadata
- Task ID: 0008
- Title: Cross-organizational 3PC deployment (mTLS transport, SSH-free launch, per-party data)
- Requested by: Adam Godel
- Owner: Adam Godel
- Date: 2026-09-09
- Status: In Progress
- Estimated effort: 3-5 days (the TLS retrofit dominates)
- Related documents: [tasks/0007](0007_dockerize-cryptdough.md), [docker/README.md](../docker/README.md)
- Branch: orchestration

## Problem Statement
- **What is the request?** Extend the Dockerization from task 0007 so a real 3PC
  computation can run across three *mutually-distrusting* computing parties: one of
  blinky/pinky/inky/clyde, plus two machines belonging to two other organizations.
- **Why does this matter now?** Task 0007 is validated, but every part of it assumes a
  single administrative domain: containers SSH into one another, one controller compiles
  and `scp`s the binary to peers, `--add-host` fakes a shared `/etc/hosts`, and the wire
  is unencrypted because the cluster sits on a trusted private subnet. None of those hold
  across organizations. Critically, `create_common_prg_among_group`
  (`include/backend/common/rand_setup.h:41-66`) sends AES seeds **in plaintext**, so an
  on-path attacker recovers every common PRG and with it the security of the computation.

## Goals
- Goal 1: Each organization runs one local command; no party needs SSH access to another.
- Goal 2: All party-to-party traffic is authenticated and encrypted (mTLS, TLS 1.3).
- Goal 3: Each party's private input never leaves its own machine.
- Goal 4: Parties provably agree on protocol configuration before launch, rather than
  discovering a mismatch as an unexplained hang.
- Goal 5: Existing benchmark numbers remain reproducible (TLS is opt-in at build time).

## Non-Goals
- Non-goal 1: Malicious security. Protocol 3 (Araki et al.) is semi-honest; see Risks.
- Non-goal 2: Replacing `run_experiment.py` for single-org benchmarking. It stays as-is.
- Non-goal 3: A PKI or CA. Three known parties are handled by certificate pinning.
- Non-goal 4: Fixing the 2PC/REAL-triples coproto issue carried over from task 0007.
- Non-goal 5: Cross-org support for the MPI communicator.

## Scope
- **In scope:** TLS in the nocopy communicator behind a `-DTLS` flag; a cross-org
  launcher, cert generation, manifest tooling and runbook under `docker/`; one worked
  example of per-party private input.
- **Out of scope:** the build system beyond the `-DTLS` option and OpenSSL linkage; all
  protocol and operator code.

## Impact Assessment
- **User impact:** additive. `-DTLS=OFF` is the default and reproduces today's behaviour.
- **Performance impact:** with `-DTLS=ON`, record encryption is on the critical path.
  AES-NI makes this fast, but on a 25 Gbps LAN it may become the bottleneck. TLS and
  non-TLS results must never be compared in the same table; re-baseline.
- **Security and privacy impact:** the point of the task. Closes the plaintext seed
  exchange and adds mutual authentication. Introduces private-key material as a new
  operational surface.
- **Backward compatibility impact:** none when `-DTLS=OFF`. `-DTLS=ON` changes the wire
  protocol, so all parties must agree.

## Context

### Relevant files
- `include/backend/nocopy_communicator/network_utils.h` — `socket_create`,
  `socket_connect`, `send_meta`/`recv_meta`, `send_wrapper`/`recv_wrapper`.
- `include/backend/nocopy_communicator/startmpc/startmpc.h` — `socket_maps`, connection
  setup, the port arithmetic.
- `include/core/communication/no_copy_communicator/no_copy_communicator.h` —
  `PartyInfoBasic::sockfd` and **four bare `recv()` calls** at lines 169, 209, 263, 354.
- `include/core/communication/no_copy_communicator/no_copy_communicator_factory.h` —
  `socketMaps_` (line 199) and the `send_wrapper` call site (line 53).
- `include/backend/common/rand_setup.h` — the plaintext seed exchange that motivates this.
- `include/backend/common/setup.h:49-82` — the `--file-args`/`-f` mechanism reused for
  the run manifest.
- `include/core/containers/tabular/encoded_table.h:492` — `inputCSVTableData`, the
  per-party private input path.
- `docker/run-party.sh` — SSH-free launcher from task 0007; the basis for the cross-org one.

### Constraints discovered
1. **Seeds are exchanged in plaintext** and the README (line 92) confirms no TLS.
2. **Four bare `recv()` calls bypass the wrappers.** A TLS retrofit that converts only
   `send_wrapper`/`recv_wrapper` will compile, run, and silently carry most MPC traffic
   in the clear. The fd must become inaccessible so the pattern cannot recur.
3. **Ports depend on `-t`:** `base_port + H*T*i + T*j`. If parties disagree on thread
   count the port arithmetic diverges and nothing connects, with no diagnostic.
4. **Party 0 needs no inbound ports at all** (it only connects to higher ranks), so the
   most network-restricted organization should be assigned rank 0.
5. **`socket_connect` retries 3 times with 1s sleeps** (~4 s), while listeners block in
   `accept()` forever. Hence descending-rank launch plus a retry supervisor.
6. **`getaddrinfo` is used for peer resolution**, so real public DNS names and IPs work
   directly; the `node0/node1` convention is only needed by `run_experiment.py`.

## Alternatives

### Option A: Secure the channel outside CryptDough (WireGuard mesh)
Pros:
- Zero code change; encrypts everything including the seed exchange.
- Handles NAT and gives stable private IPs.
Cons:
- Security depends on infrastructure each organization must deploy and maintain
  correctly; CryptDough itself still ships an insecure transport.
- A misconfigured or absent tunnel fails open, silently.

### Option B: TLS inside the nocopy communicator (chosen)
Pros:
- The framework is secure by construction; no external dependency to get wrong.
- Fails closed: a failed handshake aborts rather than falling back to plaintext.
- Mutual authentication comes with it, so parties know who they are talking to.
Cons:
- A real C++ change across four files, including the four bare `recv()` sites.
- Adds OpenSSL as a dependency and TLS cost to the critical path.

### Recommendation
- **Chosen: Option B**, gated behind `-DTLS=ON|OFF` (default OFF) so benchmark builds
  stay byte-for-byte comparable with published numbers.
- Confirmed with the requester, along with: each party builds from a pinned commit;
  per-party input via `inputCSVTableData`; descending-rank launch plus a retry supervisor.

## Implementation Plan
1. `Conn` abstraction in `network_utils.h` (fd + `SSL*`, `conn_send`/`conn_recv`,
   private fd), with a no-TLS compile path preserving today's behaviour.
2. TLS context construction: TLS 1.3 minimum, mTLS, SHA-256 certificate pinning via a
   custom verify callback.
3. Thread the `Conn` type through `startmpc.h`, `no_copy_communicator_factory.h`, and
   `no_copy_communicator.h`, converting **all four** bare `recv()` sites.
4. `CMakeLists.txt`: `-DTLS` option, `find_package(OpenSSL REQUIRED)`, link libraries.
   `docker/Dockerfile`: `libssl-dev`.
5. `docker/gen-party-certs.sh`, `docker/run-party-external.sh` (TLS material, data mount,
   retry supervisor, real hostnames), `docker/build-party.sh`, `docker/check-manifest.sh`,
   `docker/manifest.example.yaml`, `docker/DEPLOYMENT.md`.
6. `examples/ex6_three_party_private_input.cpp`.
7. Update `tasks/tasks.md`.

## Risks and Mitigations
- **Risk:** a partial retrofit leaves some traffic in plaintext.
  - **Mitigation:** make the fd private so plain `recv()` will not compile; verify with
    `tcpdump` against a `-DTLS=OFF` control that demonstrably *does* show the seed.
- **Risk:** TLS silently degrades benchmark results.
  - **Mitigation:** default OFF; re-baseline before any TLS-on measurement.
- **Risk:** certificate pinning is bypassed by a fallback path.
  - **Mitigation:** explicit test that an unpinned cert fails the handshake closed.
- **Risk:** protocol 3 is semi-honest, so "mutually distrusting" is only partly true.
  - **Mitigation:** state the threat model explicitly in `DEPLOYMENT.md`; TLS protects
    against outsiders, not against a deviating party.
- **Risk:** key material leaks via image layers or logs.
  - **Mitigation:** keys are mounted read-only at run time, never `COPY`d into the image.

## Rollback and Recovery
- **Rollback:** build with `-DTLS=OFF` (the default) to return to task 0007 behaviour;
  the new `docker/` scripts are additive and can be deleted.
- **Recovery:** certificate rotation is regenerate + redistribute fingerprints + restart;
  no state is persisted.

## Validation Plan
- **Automated:** full suite under `-DTLS=ON` and `-DTLS=OFF`; results identical.
- **Manual:** `tcpdump` proof that the seed is encrypted with TLS on and visible with it
  off; pinning rejects an impostor; three-machine run on b/p/i/c with container SSH
  disabled; iptables simulation of a rank-0 party with no inbound ports; manifest
  divergence caught pre-launch; per-party CSV input with no plaintext on the wire.
- **Success criteria:** a three-party computation completes across three organizations
  with encrypted, mutually authenticated transport and no shared SSH access.

## Definition of Done
- Suite passes with TLS on and off; the `tcpdump` contrast is demonstrated.
- A cross-org dry run with synthetic data completes end to end.
- `docker/DEPLOYMENT.md` is complete enough for another organization to follow unaided.
- Tests updated or added: TLS on/off suite runs; pinning-rejection check.
- Documentation updated: `DEPLOYMENT.md`, `docker/README.md`, main `README.md` pointer.

## Approval
- [x] Task document reviewed
- [x] Approved to implement (plan approved 2026-09-09)
- Approver: Adam Godel
- Approval date: 2026-09-09

## Change Log
- 2026-09-09: Initial draft created; design decisions confirmed with the requester.

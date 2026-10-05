# Task 0020 — Full pipeline as real WAN 3PC: two CloudLab data owners, blinky compute party

## Metadata
- Task ID: 0020
- Title: Full `mpc-analysis` run across Utah (CloudLab) and Boston (blinky), Docker + mTLS, socat egress via wormhole
- Requested by: Adam Godel
- Owner: Adam Godel
- Date: 2026-09-26
- Status: Done
- Estimated effort: 1 day (dominated by the ARM image builds and the run itself)
- Related documents: [0008](0008_cross-org-3pc-deployment.md), [0013](0013_single-owner-node-runner.md), [docker/DEPLOYMENT.md](../docker/DEPLOYMENT.md), [report.md](../report.md)
- Branch: logistic-regression @ `03f28f95efcf`

## Problem Statement
- **Request:** run the whole 3PC analysis pipeline (all 14 model fits, plus the aggregate
  nodes) with the two CloudLab machines as the data owners and blinky as the third party.
  Use Docker and TLS, and treat the two CloudLab machines as fully independent
  organizations even though they share a cluster. blinky has no public IP, so its traffic
  goes through `wormhole` (same firewall, public IP) using `socat`. Report runtime and
  accuracy, and write a concise runbook (`report.md`).
- **Why now:** task 0019 measured real 3PC on a LAN (blinky/pinky/inky, ~0.2 ms RTT). This
  is the first run across a real WAN (~53.5 ms RTT) with owners in a different
  administrative domain.

## Goals
- Each party builds its own image, generates its own keypair, and (for owners) holds only
  its own data. Nothing but public material (certificates, manifest) crosses between parties.
- No party has shell access to another. **No ssh from one node to another**: every party is
  driven by its own session from the operator's laptop.
- All MPC traffic is mTLS 1.3 end to end. The socat relays see only ciphertext.
- Per-model wall clock plus accuracy against the plaintext oracle, compared with task 0019's
  LAN run on the same data.

## Non-Goals
- Code changes to CryptDough. Findings that need code (e.g. `TCP_NODELAY`) are reported only.
- Malicious security (protocol 3 is semi-honest, as in 0008).

## Topology

| Rank | Machine | Role | Arch | Inbound needed |
|---|---|---|---|---|
| 0 | blinky.bu.edu (192-core EPYC 9655P, no public IP) | compute party, no data | x86_64 | none |
| 1 | ms0625.utah.cloudlab.us, 128.110.216.216 (8-core X-Gene 1, 62 GB) | owner A (`-pa 1`) | aarch64 | base+1 from rank 0 |
| 2 | ms0614.utah.cloudlab.us, 128.110.216.205 (same) | owner B (`-pb 2`) | aarch64 | base+2 from rank 0, base+5 from rank 1 |

blinky takes rank 0 because rank 0 needs no inbound ports. Ranks 1 and 2 dial each other on
their **public** IPs, not the 10.10.1.x experiment LAN.

### Measured network facts
- blinky↔CloudLab RTT 53.5 ms; blinky↔wormhole 0.3 ms.
- wormhole's host firewall (firewalld, no sudo) **rejects inbound high ports** from blinky
  ("No route to host"). wormhole → blinky high ports **are** open.
- wormhole had no socat and no Docker; socat 1.8.0.3 was built from source into `~/bin`.

## Alternatives for the blinky ↔ wormhole hop

### Option A: lazy reverse socat bridge (chosen)
blinky: `socat TCP-LISTEN:<app>,bind=127.0.0.1 TCP-LISTEN:<relay>,range=<wormhole>/32`.
wormhole: `socat TCP:blinky:<relay>,retry TCP:<owner>:<app>`. Rank 0 dials `127.0.0.1:<app>`.
Pros: needs nothing opened inbound on wormhole; the owner is only contacted once rank 0
really dials; plain TCP relay, so TLS stays end to end. Cons: two socat processes per
connection; rank 0's host list differs from the owners' (harmless: the list is used only to
dial higher ranks, `startmpc.h:137-146`).

### Option B: socat over an ssh stdio channel to wormhole
Pros: uses port 22, which is open. Cons: double encryption; one ssh login per MPC
connection (30+); an ssh hop is what the user wants to avoid in the data path.

### Option C: skip wormhole (blinky reaches CloudLab directly)
blinky *can* reach CloudLab high ports directly. Cons: not the requested topology; it
would not model a compute party whose only egress is the proxy.

## Plan
1. Owners (each, own session): install Docker; clone the public repo at the pinned commit;
   `build-party.sh`; `gen-party-certs.sh`; compile into `~/CryptDough/build`; generate its
   own synthetic half locally and keep only that half; `count-analysis-rows.py`;
   `check-manifest.sh`.
2. blinky: pinned source via `git archive`; compile PROTOCOL=3 TLS=ON. It reuses the
   existing `cryptdough:tls` image because blinky's root disk has 5.7 GB free, which is too
   little for a `--no-cache` rebuild. Deps are identical; the only Dockerfile change since
   that image is the Python venv stage.
3. Exchange certificates via the laptop and compare fingerprints.
4. Smoke test: one IRLS node (`6a:any`) to measure WAN overhead, then an ETA.
5. Full run: 15 concurrent jobs (`-S describe` + 14 `-N <model>`), base port 31000 + 100·k,
   launched in descending rank order.
6. Scoring on the laptop: fetch each owner's (synthetic) half as the auditor, then run
   0019's `score.py` / `timings.py`.

## Risks
- **Round-bound at 53.5 ms RTT.** 0019 found fits round-bound on a LAN, and a WAN RTT is
  ~200x longer. Mitigation: smoke-test first; run fits concurrently.
- **Mixed architectures.** The owners are aarch64 and blinky is x86_64. Share arithmetic is
  exact, but public constants computed in `double` could differ by an ULP (GCC contracts
  FMAs on aarch64 by default). Mitigation: compare accuracy with 0019's x86-only run on
  identical data.
- **No `TCP_NODELAY`** on the communicator's sockets (`network_utils.h`), so Nagle can add
  delay on the WAN leg. Mitigation: socat sets `nodelay` on the legs it owns; report it.
- **Weak owner CPUs** (X-Gene 1). The 15 jobs share 8 cores per owner.

## Findings (2026-09-26, smoke test `-N 6a:any`)

### BLOCKER: the common PRG silently produces zeros on the ARM owners
- `AESPRGAlgorithm::aesGenerateValues` (`include/core/random/prg/prg_algorithm.h:229`)
  calls libsodium's `crypto_aead_aes256gcm_encrypt_detached` and **ignores its return
  value**. libsodium 1.0.18 implements AES-256-GCM only with hardware AES. The X-Gene 1
  reports `Features: fp asimd evtstrm cpuid` (no `aes`).
- The probe, run inside each party's own image:

  | party | `aes256gcm_is_available` | return | keystream |
  |---|---|---|---|
  | ms0625 (aarch64) | 0 | -1 | `0000000000000000` |
  | ms0614 (aarch64) | 0 | -1 | `0000000000000000` |
  | blinky (x86_64) | 1 | 0 | `1565c0437ae51fb5` |

- Consequence: the parties' common-PRG streams disagree, so replicated shares do not
  reconstruct. In the smoke run the opened `patients` counts came out as random 64-bit
  values (`1633495237182776832`, …) instead of 188/115/115. Nothing reported an error.
- It is also a **security bug** independent of this run: on any CPU without AES, every
  party's "randomness" is all zeros, and an all-ARM deployment would appear to work.

### WAN cost
- Ingestion plus the relational stage took **183 s**, against ~1 s on 0019's LAN: ~180x,
  in line with the RTT ratio (53.5 ms vs ~0.3 ms). If the fits scale the same way, a
  mixed fit that took 10–31 min on the LAN would take roughly 30–90 h.
- Each job keeps one thread at ~100% CPU (busy wait), so 15 concurrent jobs would
  oversubscribe the owners' 8 cores about 2x.

### After the PRG fix (`5a8c739`, 2026-09-26/27)
- `test_randomness` passes on both ARM owners (OpenSSL path) and on blinky (libsodium
  path), including the new known-answer vector, which pins both paths to the same keystream.
- Smoke test `-N 6a:any` over the WAN: all three parties exit 0, `patients` = 188/115/115,
  and the output CSV is **bit-identical** to 0019's LAN run (estimates, SEs, z). The
  mixed-architecture run is exact.
- **Timing:** connect 0.4 s; ingest + relational 182 s (~180x the LAN); **6a fit 17,583 s
  (4.9 h) vs 14.8 s on the LAN, ~1,190x**. That is ~6.6x worse than pure RTT scaling
  (53.5 / ~0.3 ms), i.e. ~360 ms per protocol round instead of ~54 ms.
- Suspected cause: no `TCP_NODELAY` on the communicator's sockets. Nagle holds small
  follow-up writes until an ACK arrives, and Linux delays ACKs by up to 40 ms, so each
  round can absorb several stalls. The ARM cores may contribute.
- At ~1,190x, the mixed fits (607–1,870 s on the LAN) would take **8–26 days each**:
  not feasible as-is.

### After TCP_NODELAY (`255fb00`, 2026-09-27)
- `set_tcp_nodelay()` is applied on both the connecting and the accepted sockets. It passes
  `test_randomness` with TLS on and off, and a local synthetic `6a` is bit-identical to 0019.
- WAN smoke `-N 6a:any`: all parties exit 0, and the CSV is again **bit-identical** to 0019.

  | stage | LAN (0019) | WAN, no NODELAY | WAN, NODELAY |
  |---|---|---|---|
  | ingest + relational | ~1 s | 182 s | **109 s** |
  | 6a IRLS fit | 14.8 s | 17,583 s | **8,872 s** (~600x LAN) |

- Fit traffic: 94.4 MB, measured with a local byte profile of the same node. The traffic
  runs in a ring (1→0, 0→2, 2→1). Throughput varies between 0.3 and 1.2 MB/min,
  depending on the phase. That is round-bound, not bandwidth-bound.
- Still ~3x above pure RTT scaling. Remaining suspects: slow ARM cores on the critical
  path of every round, and the extra hop through the relay.
- Projection at ~600x: mixed fits (607–1,870 s on the LAN) would take **4–13 days each**.

### Owners moved to CloudLab UMass (2026-09-27)
The Utah nodes were retired at the requester's direction. New owners: pc08 (198.22.255.18,
rank 1) and pc22 (198.22.255.32, rank 2), Xeon E5-2660 v3, 40 cores, x86_64 with AES-NI.
The RTT from blinky and from wormhole is 3.5 ms. Same setup as before: each party builds
its own image at `255fb00` (fingerprint `47c69a39…`), generates its own keypair and its
own data half (hashes match 0019), and pre-flight passes.

| | LAN (0019) | Utah (53.5 ms) | UMass (3.5 ms) |
|---|---|---|---|
| ingest + relational | ~1 s | 109 s | **11 s** |
| 6a IRLS fit | 14.8 s | 8,872 s | **420 s** (~28x); CSV bit-identical to 0019 |
| mixed-model BFGS iteration (2a:umass) | ~67 s | — | **1,641 s, 1,650 s** (~24.5x) |

Full-pipeline projection: ~25x each model's 0019 LAN time. The longest fit (5a
non-UMass, 25 iterations) takes ~13 h; the 14 models back to back ~4.7 days. As 15
concurrent jobs, it finishes in ~13–14.5 h (0019 measured <=10% concurrency overhead).

### Full run launch (2026-09-27 12:52)
15 concurrent jobs: `describe` plus 13 `-N` nodes on base 31000 + 100k, and `2a:umass`
(already running from the measurement) on 33100.

- **Bug: `-cl 1` (conflict_list) is broken under `-D`.** `SecureConflictCount`
  (`playground/secure.h:443`) sizes its vectors from the owners' raw file row counts,
  which only the owner knows. Owner A computes m=512, owner B m=512, and the compute
  party, which holds neither file, m=2. The parties desynchronize: the opened count was
  `3248614333095933440`, and the job then hung. This is the "public values must not be
  read off private data" failure from DEPLOYMENT.md. Synthetic mode (0019) never hits it,
  because every party generates both halves. Not fixed here: `describe` was rerun without
  `-cl 1` (it is opt-in). All 9 aggregate rows are identical to 0019.
- `6a:any` aborted once at startup: all three parties reported `Conn::send_all: send failed`
  within 0.4 s of launch. The relay and port wiring checked out. A relaunch on the same
  ports ran cleanly. Not reproduced; logged as transient.

### Full run result (Sun 12:52 → Mon 01:58, 13.1 h wall clock)
- All 15 jobs exited 0 on all three parties. The per-model runtime and accuracy table is in
  [report.md](../report.md#results-2026-09-27-commit-255fb00-umass-owners-rtt-35-ms).
- Every model ran 21–30x its LAN time. 6a/6b are bit-identical to 0019; every 2a/2b fit is
  within 0.08 SE of the oracle; 5a/5b are comparable to the LAN run, except 5a:any, which
  stopped at a worse point (NLL excess 0.48 vs 0.02, sigma^2 0.056 vs 0.196).
- Open: the `-cl` conflict_list bug under `-D` (above).

### Options for the blocker (resolved: A, committed as 5a8c739)
- **A. Fix the PRG (code change).** Generate the keystream with OpenSSL's
  `EVP_aes_256_gcm`, which has a portable software path and is already linked for TLS.
  Its output is byte-identical to libsodium's on x86, so existing results do not move.
  Also fail closed on any error return.
- **B. Different owner hardware.** Swap the owners for x86 CloudLab nodes with AES-NI.
  No code change, but the silent-zero bug stays in the tree.
- **C. Portable cipher (ChaCha20)** for every party. It changes every existing PRG
  stream, so benchmark reproducibility is lost.

## Validation
- All 15 jobs exit 0 on all three parties; TLS enabled with 2 pinned peers on each.
- Accuracy compared with the oracle and with 0019 (same data, SHA-256 `9f68…`/`bf8f…`).

## Approval
- [x] Requested directly by Adam Godel (2026-09-26); scope as stated in the request.
- Approver: Adam Godel

## Change Log
- 2026-09-26: Draft. First attempt staged both synthetic halves from blinky; per the
  requester, the CloudLab nodes were wiped completely and the setup restarted. Each owner
  now generates its own half, and blinky never holds plaintext data.

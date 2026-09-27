# Task 0020 — Full pipeline as real WAN 3PC: two CloudLab data owners, blinky compute party

## Metadata
- Task ID: 0020
- Title: Full `mpc-analysis` run across Utah (CloudLab) and Boston (blinky), Docker + mTLS, socat egress via wormhole
- Requested by: Adam Godel
- Owner: Adam Godel
- Date: 2026-09-26
- Status: Blocked (PRG on ARM owners; see Findings)
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

### Options for the blocker (decision needed)
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

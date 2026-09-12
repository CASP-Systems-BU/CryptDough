# 0013 — Run one lineage node, for one data owner

## Metadata
- Task ID: 0013
- Title: Single-node execution and a single-owner input/output path for `mpc-analysis`
- Requested by: Adam Godel
- Owner: Adam Godel
- Date: 2026-09-11
- Status: Done
- Estimated effort: Medium (~400 lines, one new header)
- Target completion date: —
- Related issue or PR: —
- Related documents: [tasks/0009](0009_mpc-analysis-pipeline.md) (the pipeline and its 17
  terminal outputs), [tasks/0011](0011_two-owner-merge-and-oblivious-rank.md) (the two-owner
  merge this adds an alternative to), [docker/DEPLOYMENT.md](../docker/DEPLOYMENT.md);
  lineage map at <https://cs-people.bu.edu/liagos/pilot/mpc_analysis_lineage.html>
- Branch: `orchestration`

## Problem Statement
- **What is the request?** Two things.
  1. Make it possible to run **one** node of the lineage — an individual model fit, or an
     individual relational query step — rather than the whole pipeline.
  2. Support the data model where **one** party provides the single input CSV, secret-shares
     it to the other two, and is the **only** party the opened output reaches. Write that
     output in a clear, standard form chosen to match the shape of the node's result.
- **Why does this matter now?** The pipeline is a wide fan-out — every node reads a source
  table directly and nothing downstream consumes anything else (`mpc-analysis.cpp:8`) — so
  "compute one node" has always been meaningful and has never been expressible. Running 14
  fits to look at one of them costs about 40 minutes under `PROTOCOL=1`. Separately, the
  deployment currently assumes the two-owner data model of task 0011; a single-site run, and
  every run where one organisation holds the whole table, has to pretend to be two owners
  and publishes its results to parties that contributed nothing.

## Goals
- Goal 1: `-N <node>` selects one terminal output — a relational query step, an aggregate
  node, or one (specification, population) model fit — and runs only that.
- Goal 2: `--owner <party>` names a single data owner. That party holds the one input CSV,
  the others receive shares, and the node's opened output reaches that party alone.
- Goal 3: `-o <path>` writes the result in a form matched to the node: a relational step is
  a table, an aggregate node is a small table, a fit is a coefficient table. All CSV.
- Goal 4: neither the two-owner path nor the full-pipeline default changes behaviour.

## Non-Goals
- Non-goal 1: making the mixed-model **optimiser** single-party. `MinimizeBFGS` opens
  objective values and directional derivatives to steer its Armijo line search, and every
  party has to take the same branch for the loop trip counts to stay public. That leak
  predates this task (0009 records it as a follow-up) and is documented at the call site
  rather than papered over. Steps 6a/6b, fitted by IRLS, have no such intermediate.
- Non-goal 2: a node-dependency graph. The lineage is a fan-out; every node needs ingestion
  plus the relational stage and nothing else, so a scheduler would have nothing to schedule.
- Non-goal 3: changing what the full run publishes.
- Non-goal 4: an oblivious `conflict_list` under a single owner. One organisation can see its
  own MRN collisions in plaintext; the node is answered for uniformity, not for privacy.

## Scope
- In scope: `playground/output.h` (new), `playground/primitives.h` (open-to-one-party),
  `playground/secure.h` (single-owner ingestion; the pipeline split), `playground/nodes.h`
  and `playground/regression.h` (a `reveal_to` parameter), `playground/mpc-analysis.cpp`
  (CLI and dispatch), this document, the `tasks/tasks.md` ledger line.
- Out of scope: `include/`, `docker/`, `examples/`, `CMakeLists.txt` (the `playground/*.cpp`
  glob already covers the target, and no new `.cpp` is added).

## Impact Assessment
- **User impact:** additive. Every existing invocation behaves exactly as before; the five new
  flags all default to the current behaviour.
- **Performance impact:** two reductions, one addition.
  - A single-node run skips the other sixteen terminal outputs, and skips the accuracy
    harnesses unless `-S kernels` asks for them by name.
  - Single-owner ingestion pads to `NextPowerOfTwo(n)` rather than `2 * NextPowerOfTwo(n)`
    and performs no `bitonic_merge`, because one owner sorts its whole table locally. Every
    downstream sort and scan then runs over half the rows.
  - Each masked open costs one extra secret-share and one extra open. Negligible against the
    fit it terminates.
- **Security and privacy impact:** this is the point of the task, so stated precisely.
  - `SharedVector::open()` reveals to every party. `OpenRawToParty` masks first: the recipient
    shares a uniformly random `r` over the whole ring, all parties open `x + r`, and only the
    recipient subtracts `r`. `x + r` is uniform on `Z_2^64` and independent of `x`, so the
    open discloses nothing to the others — information-theoretically, not computationally.
    The mask comes from `/dev/urandom` through the framework's own PRG, not a seeded `std::`
    generator, because a recoverable generator state would recover `x`.
  - What still reaches every party in a single-owner run: the **public** parameters (row
    count, padded length, model shape, trip counts), and — for models 2a/2b/5a/5b only — the
    line-search scalars inside `MinimizeBFGS` (Non-goal 1). Steps 6a/6b and every relational
    and aggregate node open nothing but their designated output.
  - Revealing a whole relational table to the owner discloses nothing new: with a single
    owner that party supplied every input row, and the table is a deterministic function of
    them. The same output under the two-owner model would be a serious disclosure, which is
    why `-o` on a relational node is only sensible with `--owner`, and why `-Q` (the SQL
    cross-check, which opens all three tables to everybody) is refused alongside `--owner`.
- **Backward compatibility impact:** none. Signatures gained defaulted parameters;
  `RunSecurePipeline` kept its name and behaviour.

## Context
- Relevant files and modules:
  - `playground/mpc-analysis.cpp` — the driver: stage selection, ingestion, node loop.
  - `playground/secure.h:157` `MergeTwoOwners`, `:617` `RunSecurePipeline` — the two-owner
    ingestion and the sequencing stage that had to be separated from it.
  - `playground/secure.h:107` `BuildHalf`, `:139` `LocalOrder` — reused unchanged by the
    single-owner path.
  - `playground/regression.h:235` `FitLogisticIrls`, `:766` `FitGlmmLaplace`.
  - `playground/nodes.h:30` `ReportSisaCounts`.
  - `playground/harness.h:458` `CohortFromSecure` — the open-to-everybody form, kept for the
    cross-checks; `OpenCohortToParty` is the deployment counterpart.
  - `include/core/containers/shared_vector.h:143` `open()` — reveals to all parties; there is
    no open-to-one in the framework.
  - `include/core/random/prg/prg_algorithm.h:239` `DevUrandomPRGAlgorithm`.
- Dependencies: none new.
- Constraints and assumptions: `-fwrapv` is on (`CMakeLists.txt:8`), so the mask's ring
  wraparound is defined rather than UB. Loop trip counts must stay public.

## Alternatives

### Option A (open-to-one): protocol-level open, one party sends its missing share
Pros:
- Cheapest possible — in replicated 3PC the recipient already holds two of three shares, so
  one party sends one vector.
Cons:
- Protocol-specific. It would have to be written again for `plaintext_1pc`, `beaver_2pc`,
  `dalskov_4pc`, `custom_4pc` and `spdz2k_npc`, and it reaches into each protocol's share
  layout — exactly the coupling the container classes exist to prevent.

### Option B (open-to-one): masked open — recipient shares `r`, all open `x + r` (chosen)
Pros:
- Protocol-agnostic: uses only `secret_share_a`/`secret_share_b` and `open()`, so it behaves
  identically under every protocol the framework compiles, including `PROTOCOL=1`.
- Information-theoretically hiding, and the argument fits in a sentence.
Cons:
- One extra share and one extra open per result.

### Option C (open-to-one): open to all, print only on the owner
Pros:
- No code beyond a guard.
Cons:
- Does not do what was asked. Every party's process still holds the plaintext; the
  restriction is a print statement, which is not a security property.

### Option D (single-owner ingest): reuse the two-owner path with an empty second half
Pros:
- Zero new ingestion code.
Cons:
- Pays for a `bitonic_merge` against a half that is entirely pad, and pads to twice the
  necessary length, so every later sort and scan runs over 2x the rows for nothing.

### Option E (single-owner ingest): share one locally sorted table, no merge (chosen)
Pros:
- The owner's local sort already establishes the order `bitonic_merge` exists to produce, so
  the merge is not skipped as an optimisation — it has nothing left to do.
- Halves the padded row count.
Cons:
- A second ingestion function. Contained by splitting `SequenceSharedTable` out, so
  everything after ingestion is shared verbatim.

### Recommendation
- Recommended options: **B** and **E**.
- Why preferred: B is the only one of the three that is both a real security property and
  protocol-independent; E is correct by construction rather than by approximation.
- Three further decisions, settled as follows:
  1. **The `-N` grammar** (`5a:umass`, `sisa_perct_cnt_umass`, `any_system`) takes its names
     from the lineage page. `flagged_dx` is accepted as an alias for `any_system`, because
     pass 2 and that projection are the same rows — the page draws them as two nodes only
     because the projection drops columns, and nothing here drops any.
  2. **The relational CSV is emitted in the pipeline's internal row order**
     (`subject_id, data_source, encounter_dt`) rather than re-sorted to
     `(subject_id, encounter_dt)`. A relation is unordered and the re-sort would cost a pass;
     readers that care can sort.
  3. **`--owner` restricts the full run's outputs too**, not only a single node's. The data
     model is a property of the run, not of the node selection, so the two flags compose.

## Implementation Plan
1. This document + the `tasks/tasks.md` ledger line.
2. `primitives.h`: `RandomRingVector`, `OpenRawToParty` (A- and B-shared),
   `OpenToPartyDoubles`, `OpenScalarToParty`. `reveal_to < 0` means "everybody", so call
   sites carry a parameter instead of a branch.
3. `secure.h`: `ShareOneOwner`; split `RunSecurePipeline` into ingestion +
   `SequenceSharedTable`; add `RunSecurePipelineSingleOwner`; `reveal_to` on
   `SecureConflictCount`.
4. `nodes.h`, `regression.h`: thread `reveal_to` to every open that produces published
   output — the counts, `BuildDesign`'s row counts, IRLS's `beta`/`cov`, the GLMM's
   parameters and its observed-information standard errors.
5. `output.h`: the `-N` grammar, `OpenCohortToParty`, and one writer per kind of node.
6. `mpc-analysis.cpp`: the five flags, the validation, single-owner ingestion, the dispatch.
7. Run recipe comment at the top of `mpc-analysis.cpp`.

## Risks and Mitigations
- **Risk:** a party that skips a masked open deadlocks the run, because the open underneath
  is a synchronised exchange.
  - Mitigation: every helper is collective by construction — all parties reach every open and
    the non-recipients get an empty vector back. The header says so, at each entry point.
- **Risk:** a non-recipient's `FitResult` / `SisaCounts` are filled with the placeholder 0,
  which would look like a result if something printed or wrote it.
  - Mitigation: every print and every write is gated on `pID == reveal_to`; `FitLogisticIrls`
    returns early with an empty coefficient vector rather than a zero-filled one.
- **Risk:** `-Q` opens all three analysis tables to every party, which silently undoes
  `--owner`.
  - Mitigation: the combination is refused with an explanatory error, not ignored.
- **Risk:** the single-owner path's padding differs from the two-owner path's, so a bug there
  would show up as a different table rather than a crash.
  - Mitigation: verified against the two-owner run on the same rows — `sisa_perct_cnt`,
    the patient counts and the row counts agree exactly (Validation, below).

## Rollback and Recovery
- Rollback plan: `git revert`. Nothing outside `playground/` is touched and every new
  parameter is defaulted, so a revert cannot break an existing call site.
- Recovery steps: none; no on-disk format or build target changes.
- Monitoring and alerts after release: n/a — a research binary.

## Validation Plan
- Automated checks: `-S kernels` (unchanged) still passes; the build is warning-free under
  `PROTOCOL=1` and `PROTOCOL=3`.
- Manual verification:
  - Single-owner ingestion reproduces the two-owner result on the same rows.
  - Each node kind runs, prints, and writes a well-formed CSV.
  - Under 3PC, only the designated rank prints and writes; the recipient's numbers are
    correct, which is what shows the masked open round-trips.
  - Any party can be the recipient.
- Expected success criteria: identical counts between the two ingestion paths; exactly one
  party producing output under `--owner`.

## Definition of Done
- `-N` addresses every terminal output of the lineage. **Done.**
- `--owner` gives one party the input and the output. **Done.**
- `-o` writes CSV shaped to the node. **Done.**
- The default run is byte-compatible with before. **Done** — all 75 `RESULT` lines at
  `-r 200` are byte-identical to the pre-change binary; see Validation results.
- Tests updated or added: no new file under `tests/` — `playground/` is not covered there,
  matching 0009.
- Documentation updated: this document, the ledger line, the run recipe comment.

## Approval
- [x] Task document reviewed
- [x] Approved to implement
- Approver: Adam Godel
- Approval date: 2026-09-11

## Validation results

Measured on blinky, `PROTOCOL=1` unless stated. Input: the production-range CSV from
tasks/0014, 1310 rows after the pass-1 filter, held whole by one owner.

- **Single-owner ingestion == two-owner merge.** `sisa_perct_cnt_umass` over the same rows:
  36/67, 36/63, 20/34 from the single-owner path, identical to the two-owner run; 1310
  encounters and 387 patients in `any_system` from both.
- **Each node kind.** `any_system` (1310-row CSV), `sisa_perct_cnt_umass`, `6a`, `5a:umass`
  and `conflict_list` each run alone and write a well-formed CSV.
- **Argument validation.** `-N 6a:umass` is refused with the reason (`data_source` does not
  vary inside either subset); `-N nonsense` prints the node list; `-o` without `-N` is
  refused; `--owner` with `-Q` is refused.
- **3PC, output to one party.** Under `mpirun -np 3 --tag-output`, `--owner 1` produced
  output from rank 1 only and `--owner 2` from rank 2 only — one copy of the table, tagged
  `[1,1]` and `[1,2]` respectively — with identical counts in both, matching the 1PC run.
- **3PC vs 1PC.** Model 6a's `visit_num` coefficient is identical (0.0870666504); the
  intercept differs by 1.8e-4, the usual gap between the two protocols' truncation.

## Follow-ups noticed while doing this

1. **`-O` reports success when it wrote nothing.** `WriteBaseTableCsv` /
   `WriteFlaggedCsv` (`etl.h:474,491`) print "could not open ..." and return, and the caller
   then prints "wrote both owners' halves ... manifest row counts: ...". A missing output
   directory therefore looks like a successful dump, with plausible row counts. Not fixed
   here — it is outside this task — but it cost real time to diagnose.
2. **The BFGS line-search leak** (Non-goal 1) is now the only thing separating models
   2a/2b/5a/5b from the single-party guarantee 6a/6b already meet. Closing it needs the
   comparison bit, not the scalar, to be what the parties act on.

# 0012 — SQLite verification layer for the relational part of the pipeline

## Metadata
- Task ID: 0012
- Title: SQLite cross-check for `flagged_dx` pass 2 and the per-system re-sequencing
- Requested by: Adam Godel
- Owner: Adam Godel
- Date: 2026-09-11
- Status: Done
- Related documents: [tasks/0009_mpc-analysis-pipeline.md](0009_mpc-analysis-pipeline.md),
  [tasks/0011_two-owner-merge-and-oblivious-rank.md](0011_two-owner-merge-and-oblivious-rank.md)
- Branch: orchestration

## Problem Statement

The relational stage — lineage node `flagged_dx` pass 2 (the window functions)
and the `umass` / `nonumass` re-sequencing — is verified against exactly one
oracle: `SequencePlain` ([secure.h:536](../playground/secure.h#L536)), a
hand-written C++ mirror of the same SQL.

That oracle and the MPC implementation (`RunSecurePipeline`,
[secure.h:617](../playground/secure.h#L617)) were written by the same person from
the same reading of the same upstream SQL. A shared misreading of
`ROW_NUMBER() OVER w_desc`, or of the `fu_month` band boundaries, yields two
implementations that agree with each other and are both wrong. Nothing in the
tree would notice.

This was already the plan. [tasks/0009:210-211](0009_mpc-analysis-pipeline.md#L210)
lists "Cross-check ... cell by cell against SQLite, using the lineage page's SQL
verbatim" under Validation Plan. Only the Python path was ever built, and it is
now dead (see Part 2).

## Goals
- Execute the upstream window SQL *as SQL*, independent of both implementations.
- Three-way: MPC vs SQL, and `SequencePlain` vs SQL. The second is the point —
  it audits the existing oracle, today's blind spot.
- Restore `scripts/testing/validate_mpc_analysis.py`, dead since task 0011.

## Non-Goals
- `flagged_dx` pass 1 (the row-local `UPPER`/`LIKE` flag parsing), the three
  `sisa_perct_cnt` aggregate nodes, and `conflict_list`. Scope confirmed with the
  requester. `sisa_perct_cnt` is covered from the Python side instead, free.
- Any change to the MPC pipeline's behaviour or cost. This is verification only.

## The finding that shaped the design

A *literal* transcription of the upstream SQL is wrong. Upstream derives the two
visit flags from two independently-ordered windows:

```sql
CASE WHEN ROW_NUMBER() OVER w      = 1 THEN 1 ELSE 0 END AS index_visit,
CASE WHEN ROW_NUMBER() OVER w_desc = 1 THEN 1 ELSE 0 END AS final_visit
```

When a patient has two encounters on their *latest* date, SQLite's descending
window does not reverse the ascending one. Measured on sqlite3 3.51, one patient
with dates 5, 10, 20, 20, 20:

```
 rid  dt  asc_rn  desc_rn
   4   5     1       5
   1  10     2       4
   2  20     3       1    <-- desc_rn = 1 lands on the FIRST row of the tied block
   3  20     4       2
   5  20     5       3
```

`w_desc = 1` marks ascending position 3. Both C++ sources mark position 5:
`SequencePlain` tests `k + 1 == j`; the MPC path uses `LastOfGroupArith` on the
physically last row of the sorted run. A multiset comparison does not absorb
this — the multisets genuinely differ.

**Resolution.** Derive both flags from one window:
`final_visit = (ROW_NUMBER() OVER w = COUNT(*) OVER (PARTITION BY subject_id))`.
Every output column is then a function of the sorted date sequence and the row's
position `k` in it, so the result is invariant to tie order and agrees with both
C++ sources. Verified to mark position 5 on the case above.

**Consequence for the comparator.** Because the values are *position-determined*,
compare per subject, ordered by `visit_num`, element-wise, and assert `visit_num`
is exactly `1..n` within each group. That is strictly stronger than a multiset
compare — it catches a duplicated or skipped rank, which is precisely the failure
mode a wrong Brent–Kung group bit would produce — and needs no tiebreak
heuristics.

**Why this would not have been caught by luck.** Under the current generator
([etl.h:377-386](../playground/etl.h#L377)) same-patient-same-date ties occur for
~1.5% of patients, but a tie at the patient's *maximum* date — the only case that
breaks the literal transcription — occurs for ~0.16%. At `TestTwoOwnerPipeline`'s
hard-coded 60 subjects the expected count is 0.09, so a naive transcription would
have passed ~91% of runs. A tie at the *minimum* date is structurally impossible
(`gaps[0] = 0`, every later gap `>= 1`).

## Alternatives

### Option A: suppress same-date ties in the generator
Pros: a full positional compare becomes well-defined, including payload.
Cons: deletes the only coverage of `BandFollowUp`'s documented same-day-repeat
branch ([secure.h:253](../playground/secure.h#L253)). Same-day ED revisits and
transfers are common in real EHR data; a verification layer that only works on
tie-free data verifies the wrong thing.
**Rejected as the primary fix.** Worth adding later as a secondary tie-free mode
under which `strict_payload` can be switched on.

### Option B: a deterministic `row_uid` tiebreak carried into the sort
`row_uid = owner * 2^32 + local_index` would make all three sources agree on a
total order, and (because `BuildHalf` uses a stable `LocalOrder`) would reproduce
`SequencePlain`'s tie order exactly.
Pros: removes the ambiguity at the root.
Cons: adds a fourth 64-bit key to `bitonic_merge` and `bitonic_sort` — roughly
+25-33% on comparator cost in the dominant stage of ingestion — to modify the
production pipeline solely to make a test easier.
**Rejected**, with one caveat recorded below.

### Option C (chosen): tie-invariant SQL + per-subject ordered comparator
No change to the generator or the sort. See Recommendation.

### Recommendation
Option C. The invariance is a property of the algorithm, so the right move is to
write SQL that has it too, and a comparator that does not assume more determinism
than exists.

**Caveat carried forward.** The invariance covers the *sequencing* columns only.
The pairing of `visit_num` to **payload** (`sisa`, `data_source`, `newage`,
`gender`) is genuinely ambiguous under ties: two same-day rows with different
`sisa` get ranks `k` and `k+1` in an order bitonic sort picks arbitrarily. So the
MPC design matrix and `PlainDesign`'s ([reporting.h:240](../playground/reporting.h#L240))
differ on ~1.5% of patients, and the fitted coefficients differ slightly for
reasons unrelated to fixed-point error. That is a **pre-existing, previously
undocumented nondeterminism** in the regression comparison — this task does not
introduce it — but it is the one argument that would justify Option B. Revisit if
coefficient tolerances ever get tight. Hence `CompareSequencing` does not compare
payload by default.

## Design

### Part 1 — the SQLite oracle

Structurally modelled on how `~/orq` verifies TPC-H: an in-memory SQLite DB on
party 0, populated from the same plaintext vectors that get secret-shared, the
equivalent SQL run at the end, results compared. Reference:
`~/orq/bench/queries/tpch/q3.cpp`, `~/orq/bench/queries/tpch/tpch_dbgen.h:173-227`.

| File | Role |
|---|---|
| `playground/sql/sequencing.sql` | The query. **Single source of truth** |
| `playground/sqlite_oracle.h` | `class SqlOracle`; the `tpch_dbgen.h` analogue |
| `playground/harness.h` | `SeqDiff` / `CompareSequencing` / `CohortFromSecure` |

`SqlOracle::Sequence(scope)` returns a `PlainCohort` — the same type
`SequencePlain` returns — so the three-way diff is three calls to one comparator
rather than three bespoke comparisons.

The SQL lives in a file, not a string literal, because both the C++ and the
Python oracle must run *identical* text. C++ embeds it at configure time via
`file(READ)` + `configure_file`; Python reads it directly. Two hand-maintained
copies could silently diverge, which is the failure mode this task exists to
eliminate.

Schema note: **no `PRIMARY KEY` or `UNIQUE` on `(subject_id, encounter_dt)`** —
~1.5% of patients legitimately duplicate that pair and a constraint would
silently drop them. Insert with two loops, never `UNION` (which dedupes).

### Part 2 — the stale Python oracle

`scripts/testing/validate_mpc_analysis.py` has been dead since task 0011: it
reads `any_system.csv` / `umass_system.csv` / `nonumass_system.csv`, which
nothing writes any more, with a `_COLUMNS` map expecting the old bracketed
schema, and reports the removed `d1a` / `d1b` nodes. It fails at its first file
open. This is why task 0011's Results section reports only the in-C++ harnesses.

A full sweep of `docker/`, `doc/`, `README.md`, `sosp-replication/`, `scripts/`
and the C++ tree found this to be the **only** stale artifact; task 0011's
cleanup was otherwise complete.

Everything below `Cohort` operates on the dataclass rather than the file format
and still mirrors the current C++ faithfully, so the fix is a new front end, not
a rewrite: read both `flagged_dx_owner_*.csv` halves (which `-O` writes and which
carry exactly the columns the sequencing needs), run `sequencing.sql` through
Python's stdlib `sqlite3` once per scope, and populate the existing `Cohort`.

This also unlocks the *other* half of task 0009's never-executed validation step
free: with the Python oracle reading the same dump, its `sisa_counts` can be
diffed against the binary's `sisa_perct_cnt` tables without bringing those nodes
into the C++ SQLite layer.

## Impact Assessment
- **User impact:** none. Verification only; no pipeline behaviour changes.
- **Performance:** none on the MPC path. The SQLite work is party-0-only,
  post-`open()`, over a few hundred rows. The main-path check is opt-in (`-Q`).
- **Security and privacy:** the oracle needs the union of both owners' halves,
  which exists only on the synthetic path. Gated on `have_oracle`, and the skip
  is **loud** — a silent skip is how a verification layer quietly stops
  verifying.
- **Backward compatibility:** `mpc-analysis` gains a SQLite link. Guarded by
  `SQLite3_FOUND` -> `HAVE_SQLITE3` so a host without libsqlite3-dev still
  builds.

## Risks and Mitigations
- **Risk:** the SQL and the C++ drift apart over time.
  - Mitigation: one `.sql` file, embedded by CMake and read by Python. Neither
    consumer holds a copy.
- **Risk:** `assert()` from party 0 aborts one rank while the others block on a
  collective (`ASSERT_SAME` in `tests/util.h` is an `assert`, and is compiled out
  under `NDEBUG` anyway).
  - Mitigation: counters and a `*** FAIL ***` line, following the existing
    harness convention; non-zero exit from `main` after all parties finish.
- **Risk:** the check passes on runs that contain no ties, which is common at 60
  subjects, and so proves nothing about tie handling.
  - Mitigation: report `tie_groups` and `CountSameDateTies()` alongside the
    PASS line; pass `num_subjects` through instead of the hard-coded 60.
- **Risk:** a verification layer that has never been observed to fail.
  - Mitigation: mandatory negative control in the validation plan below.

## Implementation Plan
1. `playground/sql/sequencing.sql`. **Done**; validated against sqlite3 3.51 on
   a fixture covering the max-date tie and the per-scope `index_visit` split.
2. CMake: `SQLite3_FOUND` guard, `${SQLite3_INCLUDE_DIRS}` (a latent gap — the
   TPCH targets compile today only because `sqlite3.h` is on the default include
   path), `HAVE_SQLITE3`, the generated `sequencing_sql.h`, and `mpc-analysis`
   singled out of the playground glob into the `LINK_SQL` list.
3. `playground/sqlite_oracle.h`.
4. `playground/harness.h`: extract `SeqDiff` / `CompareSequencing` /
   `CohortFromSecure`; rewrite `TestTwoOwnerPipeline` around them; delete the
   `ord[]` / `data_source` tiebreak alignment, which is a half-fix that gives
   false confidence.
5. `playground/mpc-analysis.cpp`: include, pass `num_subjects`, opt-in `-Q`.
6. `scripts/testing/validate_mpc_analysis.py`: new front end; delete the
   `d1a`/`d1b` reporting and its helpers, and the unreferenced `fit_all`.
7. Documentation corrections listed under "Related cleanup".

## Related cleanup
- [mpc-analysis.cpp:4](../playground/mpc-analysis.cpp#L4) describes the pipeline
  *input* as the three cohort tables; they are now outputs of the relational
  stage.
- [secure.h:171](../playground/secure.h#L171) references `ShareCohort`, removed
  in task 0011.
- `tasks/0009` has no "superseded by 0011" pointer, and several passages read as
  current when they are not — notably `:68`, which scopes out the oblivious sort
  and merge that task 0011 went on to build.
- `tasks/tasks.md:13` still bills task 0009 as 19 lineage nodes; it is 17 after
  the `d1a`/`d1b` removal.

## Follow-ups not in scope
- `docker/check-manifest.sh` does not validate the manifest's `data:` block.
- A tie-free generator mode enabling `strict_payload` payload comparison.
- The payload-pairing nondeterminism in the regression comparison (see Caveat).

## Validation Plan
1. Build with SQLite present → `HAVE_SQLITE3` defined, `mpc-analysis` links.
2. Build with `-DCMAKE_DISABLE_FIND_PACKAGE_SQLite3=ON` → still builds, prints
   the skip line at runtime.
3. `./mpc-analysis -S kernels -r 200` → all pairwise diffs zero, `tie_groups`
   reported.
4. `./mpc-analysis -S kernels -r 2000` → at least one max-date tie occurs and the
   check still passes. This is the run that would have caught the `w_desc`
   transcription.
5. **Negative control:** perturb one band boundary, confirm non-zero `bad_fu`,
   revert.
6. `-S describe -O /tmp/dump` then the Python oracle, with and without
   `--compare`.
7. `run_experiment.py -p 3` → no rank deadlock; only party 0 prints.

## Approval
- [x] Task document reviewed
- [x] Approved to implement
- Approver: Adam Godel
- Approval date: 2026-09-11

## Decisions made during implementation
- **No bug-compatible query.** An earlier draft added
  `sequencing_upstream_bug.sql`, reproducing the upstream fault (tasks/0011
  section 4) so the corrected `fu_month` behaviour could be asserted as a measured
  delta. Removed on request: the oracle should contain only the SQL CryptDough
  means to compute. The corrected per-scope behaviour is still checked — all
  three sources compute it and the three-way comparison requires them to agree.

## Results

### Verified locally (macOS, sqlite3 3.51, Apple clang, C++20)
The full MPC build is not possible locally (no blaze or cryptoTools), so the new
code was exercised through standalone builds of the real headers against stand-in
cohort types, with `-Wall -Wextra -Wpedantic` and no warnings.

- `SqlOracle` over an 8-row fixture with a three-way tie at a patient's maximum
  date: all three scopes read back with correct column alignment, and
  `final_visit` lands on rank 5, not the first tied row. Same-date tie count
  reported correctly.
- `CompareSequencing`: tie order treated as equivalent; **all three negative
  controls fire** — a `w_desc`-style `final_visit` (bad_final = 2), a perturbed
  `fu_month` band (bad_fu = 1), and a skipped rank (bad_rank = 1), the last being
  what a multiset comparison would miss. NULL vs NULL matches, NULL vs 0 does
  not; missing subjects and ragged runs are caught.
- `validate_mpc_analysis.py` on a synthetic two-owner dump in the exact
  `WriteFlaggedCsv` format (1055 rows, 300 patients): builds all three cohorts
  through `sequencing.sql`, prints the `sisa_perct_cnt` tables, and fits model 6a.
  Its SQL-built cohorts agree **exactly** with an independent Python port of
  `SequencePlain` on every scope — 1055 / 543 / 512 rows, zero mismatches —
  despite 97 / 33 / 26 same-date tie rows.

### Verified on blinky (`/scratch/adam/CryptDough-orch`, sqlite3 3.45.1)
Changed files copied over after confirming blinky's copy matched local HEAD in
everything but comments and older task docs; originals kept in
`.bak-pre-0012/`, logs in `runs-0012/`.

| Validation step | Result |
|---|---|
| 1. Build with SQLite (`build/`, PROTOCOL=1) | `HAVE_SQLITE3` defined, header generated, built in 68s, no warnings in new code |
| 2. Build without SQLite (`build-nosql/`) | builds; prints "SQL cross-check is ABSENT"; `SequencePlain` vs MPC still PASS |
| 3. `-S kernels -r 200` | all 9 comparisons MATCH (450 / 271 / 179 rows); 1 tied (patient, date) pair |
| 4. `-S kernels -r 2000` | all 9 MATCH (4502 / 2499 / 2003 rows); 20 tied pairs |
| 5. Negative controls (in `build3`, then restored) | see below |
| 6. Python oracle on the `-O` dump | all 9 `sisa_perct_cnt` rows and the null-`fu_month` counts **identical** to the binary; `--compare` against 75 `RESULT` lines: **VALIDATION: PASS** |
| 7. `mpirun -np 3` on `build3` (PROTOCOL=3) | all MATCH, PASS, no deadlock; only party 0 prints |
| `-S describe -Q 1` (full-size SQL check) | PASS on 496 / 236 / 260 rows |
| `-S describe -D dump -Q 1` round trip | loud SKIPPED message; aggregate and pipeline tables identical to the synthetic run |

**Negative controls.** Both deliberate breakages were caught, and the first one
confirms the design finding directly:

- `final_visit` rewritten to the literal upstream `ROW_NUMBER() OVER w_desc = 1`:
  **PASSES at `-r 200`** and **FAILS at `-r 2000`** with 8 / 2 / 6
  `final_visit` mismatches in SQL vs `SequencePlain` and SQL vs MPC, while
  `SequencePlain` vs MPC still matches. So the naive transcription would have
  shipped green under the old 60-subject harness, and passing `num_subjects`
  through to `TestTwoOwnerPipeline` is what makes the check able to see it.
- A `fu_month` band value changed from 3 to 4: FAILS at `-r 200` with 59 / 37 / 19
  `fu_month` mismatches.
- After restoring `sequencing.sql` (hash re-verified) and rebuilding: PASS.

**Fixed during the blinky run**
- `validate_mpc_analysis.py` bound `?1` with a tuple. Python's `sqlite3` treats
  `?NNN` as a named parameter and 3.12 warns that this becomes an error; it now
  binds `{"1": param}`. Rerun with `-W error::DeprecationWarning`: clean.
- `scipy` was never declared although `--compare` imports it for the mixed
  models; added to `requirements.txt`. Blinky has no scipy system-wide, so the
  run used a new `/scratch/adam/venv-0012`.

**Worth watching, not caused by this task:** model 5b on `any` has a max relative
SE difference of 9.86e-2 against a 1.0e-1 tolerance — the closest margin in the
comparison. It is a property of the fits, not of the relational stage.

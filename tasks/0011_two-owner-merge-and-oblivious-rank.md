# 0011 — Full pipeline from cdrcatsse_match_pcc across two data owners

## Metadata
- Task ID: 0011
- Title: Full pipeline from `cdrcatsse_match_pcc` across two data owners, with in-MPC sequencing
- Requested by: Adam Godel
- Owner: Adam Godel
- Date: 2026-09-11
- Status: Done
- Related documents: [tasks/0009_mpc-analysis-pipeline.md](0009_mpc-analysis-pipeline.md),
  upstream lineage <https://cs-people.bu.edu/liagos/pilot/mpc_analysis_lineage.html>
  and <https://cs-people.bu.edu/liagos/pilot/pcc_sisa_lineage.html>
- Branch: orchestration

## Problem Statement

Task 0009 built the pipeline on an assumption that no longer holds: that each of
the three input tables is held in the clear by one designated party, so the sort
by `subject_id` and the visit sequencing could be done locally in plaintext
before sharing. [cohort.h:330-335](../playground/cohort.h#L330) states this
explicitly and scopes the alternative out.

The real deployment has `any_system` arriving from **two** data owners whose
records overlap on `subject_id`. No party holds the union in the clear, so the
merge and everything derived from row order must happen under MPC.

This matters now because every sequencing variable the analysis depends on is
derived from row order within a patient: `visit_num`, `index_visit`,
`final_visit`, `index_dt` and the `fu_month` follow-up bands. Upstream these come
from one SQL block (lineage node `flagged_dx`, pass 2):

```sql
ROW_NUMBER() OVER w                                      AS visit_num,
CASE WHEN ROW_NUMBER() OVER w      = 1 THEN 1 ELSE 0 END AS index_visit,
CASE WHEN ROW_NUMBER() OVER w_desc = 1 THEN 1 ELSE 0 END AS final_visit,
FIRST_VALUE(f.encounter_dt) OVER w                       AS index_dt
WINDOW w      AS (PARTITION BY f.subject_id ORDER BY f.admit_date_pcc),
       w_desc AS (PARTITION BY f.subject_id ORDER BY f.admit_date_pcc DESC)
```

## Goals
- Implement the whole lineage starting at `cdrcatsse_match_pcc`, not just the
  analysis half, with the base table split across two data owners.
- Merge the two owners' records obliviously.
- Compute the sequencing variables under MPC, correctly, on the merged table.
- Reproduce the per-system re-sequencing (lineage nodes `umass` / `nonumass`)
  without either system table being held in the clear by anyone.

## Non-Goals
- Private set intersection or identity resolution. Both owners are assumed to
  use the same `subject_id` space; the upstream MRN-to-study-ID repair (lineage
  node `pcc_mrn`) stays upstream.
- Deduplication. Confirmed with the requester: the two owners' encounter sets
  are **disjoint**, so a shared patient never has the same encounter twice.
- Hiding per-owner row counts. The manifest already publishes them
  (`--rows-any` etc. in the cross-organisational path).

## Blockers found, and how each is resolved

### 1. The port had dropped the ordering key

`kCsvHeader` was twelve columns and none was a date, so ranking had nothing to
rank by.

**Correction to an earlier reading of this.** The upstream export is not missing
the date. `EXPORT_COLS` on the lineage page is thirteen columns and its second
entry is `['encounter_dt','date','Encounter date.']`, alongside `index_dt`. It
is the CryptDough port that dropped both and added `fu_month_present` to encode
NULL. So restoring `encounter_dt` restores fidelity to the documented export
rather than widening disclosure. It is never opened — only compared inside
the sort.

### 2. There is no oblivious sort in `playground/`

`SegScan` requires rows already sorted by key; `SortBySubject` is a plaintext
`std::stable_sort`. The library has `quicksort.h`, `radixsort.h`, `merge.h` and
`sorting.h`, but `playground/` imports none of them.

**Options considered:**

| | Approach | Cost | Notes |
|---|---|---|---|
| A | Concatenate both halves, full `bitonic_sort` on `{subject_id, encounter_dt}` | `log N·(log N+1)/2` compare-swap stages | Assumes nothing about the inputs. For N=2^16, 136 stages. |
| B | Each owner sorts its own half locally in plaintext, then `bitonic_merge` | `log N` stages | Merge's contract is exactly "two already sorted halves". For N=2^16, 16 stages — ~8.5x cheaper. |
| C | `EncodedTable::sort(...)` with `RADIXSORT` | — | Would mean porting `playground/` from raw `AV`/`BV` columns to `EncodedTable`. Large, unrelated restructure. |

**Recommendation: B.** Each owner holds its own half in the clear, so sorting it
locally is free and reveals nothing that owner does not already know — the same
argument [cohort.h:330-335](../playground/cohort.h#L330) already makes for the
single-owner sort. It preserves the existing design philosophy and costs a
`log N` factor less than a full sort.

**Library bug to route around.** The by-value overload of `bitonic_merge`
([merge.h:245](../include/core/operators/merge.h#L245)) pushes `_columns` into
`res` three times and leaves `_data_a_` / `_data_b_` empty, so it silently drops
every data payload and triplicates the key list. Use the pointer overload
([merge.h:168](../include/core/operators/merge.h#L168)) directly. Worth fixing
upstream separately; out of scope here.

### 3. Provenance of `umass_system` / `nonumass_system`

These are partitions of `any_system` by `data_source`. If nobody holds
`any_system` in the clear, nobody can hold its partitions either.

**Decision (requester): derive both in-MPC.**

The subtlety is that `{subject_id, data_source}` runs are **not contiguous**
after a `(subject_id, encounter_dt)` sort — a patient's encounters interleave
between systems — and the segmented scans require adjacency. So this needs a
second sort, on `{subject_id, data_source, encounter_dt}`. Using `encounter_dt`
as the third key reproduces date order inside each `(patient, system)` run
without needing a stable sort, which matters because bitonic sort is not stable.
(An earlier draft proposed `visit_num` as the third key; `encounter_dt` is
equivalent here and avoids needing an arithmetic-to-boolean conversion of the
freshly computed rank.)

### 4. The upstream ETL bug

`DeriveSystemCohort` deliberately reproduces the fault documented on the lineage
page: the ETL guards `fu_month<sfx> = 0` on the **global** `index_visit` rather
than `index_visit<sfx>`, so a patient whose history in a system starts later
than their overall history gets NULL on their first row there.

Computing the sequencing in MPC naturally produces the *correct* per-scope flag.

**Decision (requester): compute it correctly.** `SequencePlain` in `secure.h`
is the matching oracle and guards on the per-scope `index_visit` too.

## Design

### Ingestion

```
owner A half (nA rows)        owner B half (nB rows)
  sorted locally by             sorted locally by
  (subject_id, encounter_dt)    (subject_id, encounter_dt)
  padded to m                   padded to m            m = NextPowerOfTwo(max(nA,nB))
          \                        /
           \                      /
        concatenated into N = 2m rows via slice() writes
                     |
          bitonic_merge on {subject_id, encounter_dt}
          carrying every data column as payload
                     |
              merged any_system
```

Both halves pad to the same `m` because `bitonic_merge` requires equal halves.
Pad rows carry `kKeySentinel` in `subject_id` and `encounter_dt` so they sort
last under both keys and form one contiguous sentinel block at the end.

### Sequencing — the rank operator

Implemented in [segmented.h](../playground/segmented.h) and already verified:

| SQL | MPC |
|---|---|
| `ROW_NUMBER() OVER w` | `SegRank(keys, valid, Forward)` |
| `ROW_NUMBER() OVER w_desc = 1` | `LastOfGroup` |
| `ROW_NUMBER() OVER w = 1` | `FirstOfGroup` |
| `FIRST_VALUE(dt) OVER w` | `SegFirstValue(keys, dt, first)` |
| `SUM(x) OVER w_cum` | `SegScan(keys, x, Forward)` |

The rank is a segmented prefix sum over a column of ones, evaluated on the
Brent–Kung network already in `BuildScanPlan`: `ScanDepth(n) = 2·bit_width(n−1) − 1`
levels, so **O(log n) rounds**, one batched multiply per level, no comparisons
and no opens. With a cached `ScanPlan` the per-level group bits are paid once.

Direction matters and is the easiest thing to get backwards: **Forward** gives
the earliest visit seqno 1 (`ROW_NUMBER() OVER w`, what the analysis wants);
Reverse gives the last visit seqno 1 and the first row the group size. They are
related by `rank_fwd + rank_rev = m + 1`.

### `fu_month`

`gap = encounter_dt − index_dt`, then three range indicators built from `gtez`:

```
fu = 0 if index_visit;  1 if 1<=gap<=30;  3 if 31<=gap<=91;  6 if 92<=gap<=182
fu_present = index_visit | in1 | in3 | in6
```

Guarded on the per-scope `index_visit`, i.e. the corrected behaviour.

### Per-system re-sequencing

Second sort on `{subject_id, data_source, visit_num}`, then rank on the compound
key `{subject_key, ds_key}`. One pass produces the sequencing for *both* systems
simultaneously; the two cohorts differ only in which rows their `valid` mask
admits.

Note `BuildScanPlan(const BV& key)` takes a single key column, so the compound
key falls back to the uncached `aggregate()` path. Extending the plan builder to
multiple key columns is a follow-up, not a blocker.

## Impact Assessment

- **Cost.** Adds one `bitonic_merge` (`log N` stages) plus one `bitonic_sort`
  (`log N·(log N+1)/2` stages) to ingestion. Ranking itself is cheap.
- **Row counts.** All three cohorts now carry all N rows, with scope selected by
  mask, rather than three separately-sized tables. The per-system model fits
  therefore run over roughly 2x the rows they did. Oblivious compaction after
  the second sort would recover this; deferred as a follow-up.
- **Schema.** `encounter_dt` added to `kCsvHeader`; existing CSVs need
  regenerating.
- **Behaviour.** `fu_month` now correct rather than bug-compatible; plaintext
  oracle updated to match.

## Which nodes need MPC at all

The load-bearing observation, added when the scope grew to the whole lineage.
Each ETL node is either row-local or cross-party:

| Node | Operation | Cross-party? |
|---|---|---|
| `cdrcatsse_match_pcc` | base table | split across the two owners |
| `conflict_list` | `pat_mrn` -> `COUNT(DISTINCT subject_id) > 1` | **yes** |
| `pcc_mrn` | hard-coded MRN repair | row-local |
| `flagged_dx` pass 1 | `visit_type` filter, `UPPER`+`LIKE` flags, facility -> `data_source` | row-local |
| `flagged_dx` pass 2 | the window functions | **yes** |
| `umass` / `nonumass` | the same windows after a `data_source` filter | **yes** |
| `any_system` etc. | projections | row-local |

Every output column of a row-local node is a function of ONE input row, so each
owner computes it over its own half in the clear. That is what keeps every
`LIKE '%...%'` and the nine-way facility-name comparison out of MPC, where
string matching would otherwise dominate the cost of the entire pipeline.
`etl.h` is that half; `secure.h` is the rest.

`conflict_list` is a dead end upstream — nothing reads it. Here it stops being
one: an MRN can carry one study ID at each owner, and neither owner can see the
disagreement alone.

## Results

All verified on blinky (`/scratch/adam/CryptDough-orch`). The local checkout
cannot build — blaze and cryptoTools are absent and `build/` is unconfigured.

- `TestSegmented` extended with the rank checks: ragged groups (3,2,1,4), both
  directions, the cached network against `aggregate()`, and `SegFirstValue`.
  **SEGMENTED HELPERS: PASS**
- `TestTwoOwnerPipeline`, the new end-to-end harness. Generates a base table,
  splits it, runs pass 1 at each owner, merges and sequences under MPC, then
  compares against `SequencePlain` over the union:

      rows: merged 145, plaintext union 145  MATCH
      visit_num mismatches   0
      index_visit mismatches 0
      final_visit mismatches 0
      fu_month mismatches    0
      umass rows: merged 99, plaintext 99
      umass per-patient visit totals wrong: 0
      TWO-OWNER PIPELINE: PASS

- `conflict_list` against a plaintext oracle over the union of what the owners
  hold, at induced conflict rates of 0 / 5 / 15 percent: 0/0, 4/4, 8/8, all
  MATCH. The oracle has to be the post-split union, not the pre-split base
  table — the disagreement does not exist until the split creates it, which is
  the point of the node.
- The three `sisa_perct_cnt` tables and all 14 model fits run to completion,
  about 78s at `-r 200`.

## Follow-ups not done

- The by-value overload of `bitonic_merge`
  ([merge.h:245](../include/core/operators/merge.h#L245)) is broken as described
  above. Worth fixing upstream; this code uses the pointer overload.
- All three cohorts carry all N rows with scope selected by a mask, so the
  per-system fits run over roughly twice the rows they need. Oblivious
  compaction after the second sort would recover that.
- (Done after the first pass) The CSV ingestion path was reworked. See
  "Data import" below.

## Data import

The old `--data-dir` read `any_system.csv`, `umass_system.csv` and
`nonumass_system.csv` — three finished analysis tables, each held in the clear
by one party. That is exactly the assumption this task removes: those files
carry `visit_num`, `index_visit`, `final_visit` and `fu_month`, and no single
owner can compute any of them. The path had already gone dead — `LoadOwnedCohort`
was no longer called, so `-D` silently accepted a directory and generated
synthetic data anyway. It is removed, along with `MakeSyntheticCohort`,
`DeriveSystemCohort`, `ReadCohortCsv` / `WriteCohortCsv`, `PlaceholderCohort`,
`LoadOwnedCohort`, `ShareCohort`, `kCsvHeader` and `PlainCohort::SortBySubject`
(421 lines of `cohort.h`).

The import is now at the top of the lineage, one file per owner:

    -O /tmp/dump    writes base_owner_a.csv and base_owner_b.csv, each owner's
                    half of cdrcatsse_match_pcc with the diagnosis text intact,
                    and prints the manifest row counts
    -D /tmp/dump    reads them back
    -ra N -rb N     the manifest counts, AFTER the pass-1 visit_type filter
    -pa 0 -pb 1     which party owns which half
    -fa / -fb       override the file names

Each party opens only its own file; pass 1 then runs locally over it. The row
counts are public because both halves pad to one common length, and a party
cannot read the length of a file it may not see. A mismatch between a file and
its declared count is fatal rather than silent: the parties would otherwise pad
differently and the shares would not line up.

Two consequences worth stating:

- **The counts are post-filter.** `flagged_dx` pass 1 drops every encounter that
  is not Emergency or Acute Care, so the manifest declares what survives, not
  the file's line count. `-O` prints the right numbers.
- **The plaintext oracle does not exist in a real run.** It needs the union of
  both halves, which is what no party holds. The regression oracle and the
  `conflict_list` cross-check are therefore gated to the synthetic path; under
  `-D` the pipeline reports its own results and says the cross-check is absent.

A one-process build (`PROTOCOL=1`, where `getPartyID()` is always 0) stands in
for every owner, so it opens both files; the ownership guard applies once there
is more than one party. Verified: `-O` then `-D` reproduces the synthetic run
byte-for-byte apart from the generating-parameters line, which has no meaning
when the input is real data.

## Deployment documentation

`docker/DEPLOYMENT.md` and `docker/manifest.example.yaml` now describe only the
two-owner layout: the data staging, the post-filter row-count rule, the `conflict_list`
disclosure, and the fact that no plaintext cross-check is possible in a real run. The
per-table CSV descriptions were **replaced**, not annotated, so nothing in the
deployment docs refers to a layout the code no longer has. The build and launch
examples name `mpc-analysis`, and the manifest's `runtime_args` and `data` blocks
describe that run.

`docker/count-analysis-rows.py` is new. The manifest needs each owner's post-pass-1 row
count, but `mpc-analysis` connects to its peers during startup and the counts are needed
*before* the ports are agreed — so the count cannot come from the binary itself. The
script applies the pass-1 visit-type filter to one owner's half and prints the number,
with no network access. It returns 240 / 256 on the dumped halves, matching what `-O`
reports, and `--verbose` shows 290 raw rows against 240 kept.

`docker/check-manifest.sh` needed a fix to go with this. It counted `- rank:` lines
anywhere in the manifest, so the new `data.owners` list inflated the party count from 3
to 5 and printed the wrong inbound port ranges — the one thing that script exists to
get right. The count is now scoped to the top-level `parties:` block (comment-tolerant),
and the owners list uses `owner_rank:` so the two cannot be confused. Verified: the
example manifest reports `num_parties=3` and ports matching `base + H*T*i + T*j`.

## Related change

The two gray (`slate`) lineage nodes — `d1a` (visits per patient) and `d1b`
(demographics), the printed descriptive reports — were removed on request, along
with the helpers used solely by them. Seventeen terminal outputs remain.

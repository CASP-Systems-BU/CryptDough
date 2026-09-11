-- Lineage node `flagged_dx`, pass 2: the window functions, plus the `umass` /
-- `nonumass` per-system re-sequencing.
--
-- THIS FILE IS THE SINGLE SOURCE OF TRUTH FOR THE SEQUENCING SQL.
--   - C++  : embedded at configure time into build/generated/sequencing_sql.h
--            and run by playground/sqlite_oracle.h (SqlOracle::Sequence).
--   - Python: read directly by scripts/testing/validate_mpc_analysis.py.
-- Both oracles therefore execute identical text. Do not inline a copy anywhere.
--
-- Parameter ?1 is the scope: 0 = any_system, 1 = umass_system, 2 = nonumass_system.
-- Applying the scope filter BEFORE the window is what produces the per-system
-- re-sequencing: a patient's third encounter overall may be their first inside
-- one system. It is also what makes fu_month guard on the PER-SCOPE index_visit,
-- i.e. the corrected behaviour CryptDough implements (tasks/0011, section 4).
--
-- ---------------------------------------------------------------------------
-- Transcribed from the lineage page's window block:
--
--     ROW_NUMBER() OVER w                                      AS visit_num,
--     CASE WHEN ROW_NUMBER() OVER w      = 1 THEN 1 ELSE 0 END AS index_visit,
--     CASE WHEN ROW_NUMBER() OVER w_desc = 1 THEN 1 ELSE 0 END AS final_visit,
--     FIRST_VALUE(f.encounter_dt) OVER w                       AS index_dt
--     WINDOW w      AS (PARTITION BY f.subject_id ORDER BY f.admit_date_pcc),
--            w_desc AS (PARTITION BY f.subject_id ORDER BY f.admit_date_pcc DESC)
--
-- with ONE deliberate departure: `final_visit` is NOT derived from `w_desc`.
--
-- `w_desc` is a second, independently ordered window, and SQLite does not
-- guarantee its tie order is the reverse of `w`'s -- it is not. For a patient
-- with dates 5, 10, 20, 20, 20 (sqlite3 3.51):
--
--     rid  dt  asc_rn  desc_rn
--       4   5     1       5
--       1  10     2       4
--       2  20     3       1     <-- desc_rn = 1 lands on the FIRST tied row
--       3  20     4       2
--       5  20     5       3
--
-- so `w_desc = 1` marks ascending position 3. Both CryptDough implementations
-- mark position 5: SequencePlain (secure.h) tests `k + 1 == j`, and the MPC path
-- uses LastOfGroupArith on the physically last row of the sorted run. Deriving
-- `final_visit` from `visit_num = COUNT(*) OVER p` instead makes every output
-- column a function of the sorted date sequence and the row's position in it,
-- so the result is invariant to tie order and agrees with both.
-- ---------------------------------------------------------------------------

WITH scoped AS (
    SELECT * FROM flagged_dx WHERE ?1 = 0 OR data_source = ?1
),
ranked AS (
    SELECT s.subject_id, s.encounter_dt, s.data_source,
           s.newage, s.gender, s.hispanic, s.sisa, s.sa,
           ROW_NUMBER() OVER w                AS visit_num,
           COUNT(*)     OVER p                AS n_visits,
           -- Under the default RANGE UNBOUNDED PRECEDING frame this is the
           -- partition's first row, i.e. the minimum date, regardless of tie
           -- order -- provably identical to MIN(encounter_dt) OVER p. Kept in
           -- FIRST_VALUE form to stay faithful to the upstream text.
           FIRST_VALUE(s.encounter_dt) OVER w AS index_dt
    FROM scoped s
    WINDOW w AS (PARTITION BY s.subject_id ORDER BY s.encounter_dt),
           p AS (PARTITION BY s.subject_id)
)
SELECT subject_id,
       encounter_dt,
       newage,
       gender,
       hispanic,
       CASE WHEN visit_num = 1        THEN 1 ELSE 0 END AS index_visit,
       visit_num,
       CASE WHEN visit_num = n_visits THEN 1 ELSE 0 END AS final_visit,
       -- BETWEEN is inclusive in SQLite and matches BandFollowUp's gtez pairs.
       -- A genuine SQL NULL, not a sentinel: fu_month_present = 0 downstream.
       CASE
           WHEN visit_num = 1                               THEN 0
           WHEN encounter_dt - index_dt BETWEEN 1   AND 30  THEN 1
           WHEN encounter_dt - index_dt BETWEEN 31  AND 91  THEN 3
           WHEN encounter_dt - index_dt BETWEEN 92  AND 182 THEN 6
           ELSE NULL
       END AS fu_month,
       data_source,
       sisa,
       sa
FROM ranked
-- The WINDOW clause orders the window computation, not the result set.
ORDER BY subject_id, visit_num;

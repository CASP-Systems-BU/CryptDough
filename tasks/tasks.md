# Task Ledger

Short list of all tasks so far.

- 0001 — Plaintext mixed-effects logistic regression (Laplace + quasi-Newton) — Status: Done — [tasks/0001_mixed-effects-logistic-regression-laplace.md](0001_mixed-effects-logistic-regression-laplace.md)
- 0002 — Taylor-series approximations for exp/log/log1p with compile-time toggle — Status: Done — [tasks/0002_taylor-series-math-approximations.md](0002_taylor-series-math-approximations.md)
- 0003 — Robust conditional-mode Newton + BFGS line search (fix inner-Newton overflow/NaN) — Status: Done — [tasks/0003_robust-conditional-mode-and-line-search.md](0003_robust-conditional-mode-and-line-search.md)
- 0004 — Factor out logistic-regression value type — Status: Done — [tasks/0004_factor-out-logistic-regression-value-type.md](0004_factor-out-logistic-regression-value-type.md)
- 0005 — Add a simulated fixed-precision value type — Status: Done — [tasks/0005_simulated-float-value-type.md](0005_simulated-float-value-type.md)
- 0006 — Make Taylor math compatible with SimulatedFloat — Status: Done — [tasks/0006_simulated-float-taylor-math.md](0006_simulated-float-taylor-math.md)
- 0007 — Dockerize CryptDough for portable experiment execution — Status: Done — [tasks/0007_dockerize-cryptdough.md](0007_dockerize-cryptdough.md)
- 0008 — Real 3PC across three organizations (mTLS, SSH-free launch, per-party data) — Status: Done — [tasks/0008_cross-org-3pc-deployment.md](0008_cross-org-3pc-deployment.md)
- 0009 — MPC analysis pipeline: full SISA acute-care port (17 terminal outputs after 0011, 14 model fits) — Status: Done — [tasks/0009_mpc-analysis-pipeline.md](0009_mpc-analysis-pipeline.md)
- 0010 — Secure matrix inversion via Newton–Schulz iterations — Status: Done — [tasks/0010_secure-matrix-inversion-newton-schulz.md](0010_secure-matrix-inversion-newton-schulz.md)
- 0011 — Full pipeline from cdrcatsse_match_pcc across two owners: oblivious merge + in-MPC rank/sequencing — Status: Done — [tasks/0011_two-owner-merge-and-oblivious-rank.md](0011_two-owner-merge-and-oblivious-rank.md)
- 0012 — SQLite cross-check for the relational stage (`flagged_dx` pass 2 + per-system re-sequencing); restore the stale Python oracle — Status: Done — [tasks/0012_sqlite-relational-verification.md](0012_sqlite-relational-verification.md)
- 0013 — Run one lineage node, for one data owner: `-N` node selection, single-owner ingest, output opened to that party alone — Status: Done — [tasks/0013_single-owner-node-runner.md](0013_single-owner-node-runner.md)
- 0014 — Convergence over the production data ranges: magnitude is free, the `visit_num` tail is not — Status: Done — [tasks/0014_production-range-convergence-test.md](0014_production-range-convergence-test.md)
- 0015 — Unify the duplicated playground operator libraries: one copy of every operator under `playground/library/`; 4.33x slower and a coefficient sign flip from the 3-term series, both flagged — Status: In Progress — [tasks/0015_unify-duplicated-playground-operators.md](0015_unify-duplicated-playground-operators.md)
- 0016 — Consistent iteration budgets: BFGS cap 12→60 + inner Newton 3→5; no fit hits the cap now and 4 of 6 improve 12–74x; `kIrlsIterations` and `kMaxSeriesTerms` both tested and rejected; 5a/5b [all systems] still poorly converged — Status: Done — [tasks/0016_consistent-iteration-budgets.md](0016_consistent-iteration-budgets.md)
- 0017 — Supersede the pipeline's operators with the library's: pipeline onto `MinimizeBFGSBatched` and the batched inference, `MinimizeBFGS` deleted — Status: In Progress — [tasks/0017_supersede-pipeline-operators-with-library.md](0017_supersede-pipeline-operators-with-library.md)
- 0018 — Secure Cholesky inverse + oblivious condition estimate: replaces NewtonSchulzInverse, takes the inference path from 2*dim² opened values to 1+p; analytic-gradient Hessian brings all 12 mixed models inside the oracle's SE bar (VALIDATION: PASS) — Status: Implemented, tolerances still to re-measure — [tasks/0018_secure-cholesky-inverse.md](0018_secure-cholesky-inverse.md)

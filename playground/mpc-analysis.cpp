// MPC port of the SISA acute-care analysis pipeline.
//   https://cs-people.bu.edu/liagos/pilot/mpc_analysis_lineage.html
//
// Two owners' halves of cdrcatsse_match_pcc in; seventeen terminal outputs out --
// three one-row aggregate tables and fourteen fitted regression models (six
// specifications across three populations, minus the four interaction fits that
// can only run on the pooled table). The three analysis tables (any_system,
// umass_system, nonumass_system) are OUTPUTS of the relational stage, not
// inputs: no single owner can compute the sequencing columns they carry, which
// is what task 0011 exists to fix. Nothing downstream consumes anything else:
// the pipeline is a wide fan-out, and every node reads a source table directly.
//
// This file holds only the driver: stage selection, ingestion, and the loop over
// the nodes. Everything it calls lives in the headers alongside it.
//
// Design and decisions: tasks/0009_mpc-analysis-pipeline.md
//
// Layout, in dependency order:
//   primitives.h   types, constants, Clone; ClampAbs/ClampRange/Abs,
//                  RecipSeeded/Div/Recip, Sqrt/Rsqrt, Exp/Log/Log1p,
//                  Sigmoid/LogOnePlusExp; share and open helpers
//   segmented.h    SegScan/SegTotal and the cached-group-bit
//                  SegScanPlanned/SegTotalPlanned; First/LastOfGroup;
//                  CountDistinct
//   linalg.h       Gram; Cholesky factor / solve / SymmetricInverse
//   optimizer.h    matrix helpers and the Newton-Schulz inverse; BFGS with a
//                  value-plus-gradient objective and an Armijo line search
//   cohort.h       plaintext and secret-shared row layouts; synthetic
//                  generator; per-party CSV ingestion
//   nodes.h        sisa_perct_cnt x3
//   regression.h   design matrices; IRLS (steps 6a/6b); flat ragged-cluster
//                  Laplace mixed model with an analytic gradient
//   reporting.h    coefficient tables and the plaintext IRLS oracle
//   harness.h      accuracy harnesses and the cost breakdown
//
// Conventions relied on throughout, spelled out in primitives.h:
//   - Values are raw scaled integers at precision 0, rescaled by hand.
//   - `AV a = b` is a shallow copy; use Clone() before mutating.
//   - BSharedVector::operator/ (circuits.h:39) is 64 sequential rounds and must
//     not appear on a hot path; Div/Recip/RecipSeeded exist to avoid it.
//
// Run:
//   ../scripts/run_experiment.py -p 3 -r 200 mpc-analysis
//   ./mpc-analysis -S kernels                 # accuracy harnesses only (fast)
//   ./mpc-analysis -S describe -r 200         # ingestion + aggregate nodes
//   ./mpc-analysis -S models -r 200           # the fourteen fits
//   ./mpc-analysis -S bench -r 200            # cost breakdown of one fit
//   ./mpc-analysis -S models -r 200 -C 0      # ... with uncached segmented scans
//   ./mpc-analysis -O /tmp/dump               # dump both owners' halves of
//                                             # cdrcatsse_match_pcc, and print the
//                                             # manifest row counts to use with -D
//   ./mpc-analysis -D /data -ra N -rb N       # read those halves back; each party
//                                             # opens only its own file, so the row
//                                             # counts come from the manifest
//   ./mpc-analysis -pa 0 -pb 1                # which party owns which half
//   ./mpc-analysis -cc 50 -ic 5 -cl 1         # even split, 5% ID conflicts, and
//                                             # run the conflict_list check

#include "cdough.h"

#include "./harness.h"
#include "./reporting.h"
#include "./secure.h"

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;
using namespace cdough::regression;


// =============================================================================
// The pipeline
// =============================================================================

// The six specifications, fitted to the three populations. The two interaction
// models can only run on any_system, because that is the only table in which
// data_source varies -- which is exactly why the script is built around it.
std::vector<ModelSpec> AllModelSpecs() {
    std::vector<ModelSpec> out;
    const std::vector<SystemScope> scopes = {SystemScope::UMass, SystemScope::Any,
                                             SystemScope::NonUMass};

    struct Base {
        const char* step;
        TimeAxis time;
        bool covars;
        bool interaction;
    };
    const std::vector<Base> bases = {
        {"2a", TimeAxis::VisitNum, false, false}, {"2b", TimeAxis::FuMonth, false, false},
        {"5a", TimeAxis::VisitNum, true, false},  {"5b", TimeAxis::FuMonth, true, false},
        {"6a", TimeAxis::VisitNum, true, true},   {"6b", TimeAxis::FuMonth, true, true},
    };

    for (const Base& b : bases) {
        for (SystemScope sc : scopes) {
            if (b.interaction && sc != SystemScope::Any) continue;
            ModelSpec m;
            m.step = b.step;
            m.time = b.time;
            m.covars = b.covars;
            m.interaction = b.interaction;
            m.random_intercept = !b.interaction;
            m.scope = sc;
            // The SAS comments hispanic out of model 5b on the UMass subset,
            // citing sparse data.
            m.drop_hispanic = (m.step == "5b" && sc == SystemScope::UMass);
            out.push_back(m);
        }
    }
    return out;
}

const SecureCohort& PickCohort(const SecureCohort& any, const SecureCohort& umass,
                               const SecureCohort& nonumass, SystemScope s) {
    if (s == SystemScope::UMass) return umass;
    if (s == SystemScope::NonUMass) return nonumass;
    return any;
}

const PlainCohort& PickPlain(const PlainCohort& any, const PlainCohort& umass,
                             const PlainCohort& nonumass, SystemScope s) {
    if (s == SystemScope::UMass) return umass;
    if (s == SystemScope::NonUMass) return nonumass;
    return any;
}


int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();

    // Which part of the program to run. The kernel and segmented-scan harnesses
    // are fast; the model fits are not, so they are separately selectable.
    //   kernels  - fixed-point, segmented-scan and linear-algebra accuracy only
    //   describe - ingestion plus the aggregate nodes
    //   bench    - cost breakdown of one mixed-model fit
    //   models   - the fourteen regression fits
    //   all      - everything (default)
    const std::string stage = engine.getArg<std::string>("stage", "S", "all");
    const bool run_kernels = (stage == "kernels" || stage == "all");
    const bool run_describe = (stage == "describe" || stage == "models" || stage == "all");
    const bool run_models = (stage == "models" || stage == "all");
    const bool run_bench = (stage == "bench");
    const bool print_describe = (stage == "describe" || stage == "all");

    // Number of patients in the synthetic cohort. Ignored in CSV mode.
    const int num_subjects = engine.getArg<int>("subjects", "r", 200);
    // Directory holding the two owners' halves of cdrcatsse_match_pcc. Empty
    // means "use the synthetic generator". Each party opens only its own file.
    const std::string data_dir = engine.getArg<std::string>("data-dir", "D", "");
    const std::string file_a = engine.getArg<std::string>("file-a", "fa", "base_owner_a.csv");
    const std::string file_b = engine.getArg<std::string>("file-b", "fb", "base_owner_b.csv");
    // Post-filter row counts, agreed in the run manifest. Required in CSV mode:
    // both halves are padded to one public length, and a party that does not own
    // a half cannot read its length from a file it is not allowed to see. These
    // are the counts AFTER the flagged_dx pass-1 visit_type filter, which is the
    // number of rows the owner actually contributes.
    const int rows_a = engine.getArg<int>("rows-a", "ra", 0);
    const int rows_b = engine.getArg<int>("rows-b", "rb", 0);
    // Where to dump the synthetic cohort for the R / Python / SQLite cross-checks.
    const std::string out_dir = engine.getArg<std::string>("out-dir", "O", "");
    // The two data owners. Each holds part of cdrcatsse_match_pcc in the clear
    // and neither holds the union, so the merge and every window function run
    // under MPC (tasks/0011).
    const int party_a = engine.getArg<int>("party-a", "pa", 0);
    const int party_b = engine.getArg<int>("party-b", "pb", 1);
    // How strongly a patient's encounters concentrate at one owner. 1.0 gives
    // each patient entirely to one side -- no overlap, and the merge is
    // pointless; 0.5 splits every patient's history evenly.
    const double concentration =
        static_cast<double>(engine.getArg<int>("concentration", "cc", 70)) / 100.0;
    // Fraction of patients carrying a DIFFERENT study ID at the second owner.
    // This is what conflict_list exists to find and what neither owner can see.
    const double id_conflict_rate =
        static_cast<double>(engine.getArg<int>("id-conflicts", "ic", 0)) / 100.0;
    // Run the cross-party conflict_list check. Off by default: it needs its own
    // ordering, because MRN runs are not contiguous in a subject-sorted table.
    const bool check_conflicts = engine.getArg<int>("conflict-list", "cl", 0) != 0;
    // Re-run the relational stage's SQL cross-check at the production row count.
    // Off by default: -S kernels already covers the same code path, and this
    // costs three extra full-cohort opens on every describe/models run.
    const bool check_sql = engine.getArg<int>("check-sql", "Q", 0) != 0;
    // 0 routes the segmented scans through aggregators::aggregate instead of the
    // cached per-level group bits, for A/B measurement.
    g_use_cached_scans = engine.getArg<int>("cached-scans", "C", 1) != 0;

    if (run_kernels) {
        TestNewKernels(engine, pID);
        TestSegmented(engine, pID);
        // -r drives this too: the max-date ties that distinguish a correct
        // final_visit from the naive `ROW_NUMBER() OVER w_desc = 1` reading
        // occur in ~0.16% of patients, so a fixed 60 almost never sees one.
        TestTwoOwnerPipeline(engine, pID, static_cast<size_t>(num_subjects), party_a,
                             party_b);
        TestLinearAlgebra(engine, pID);
    }
    if (!run_describe && !run_bench) return 0;

    // ---------------------------------------------------------------- ingest
    //
    // The full lineage, starting at cdrcatsse_match_pcc:
    //
    //   cdrcatsse_match_pcc   generated, then split across two owners
    //   pcc_mrn               MRN repair          -- row-local, at each owner
    //   flagged_dx pass 1     filter + flags      -- row-local, at each owner
    //   conflict_list         cross-party, under MPC
    //   flagged_dx pass 2     cross-party, under MPC
    //   umass / nonumass      cross-party, under MPC
    //   the three *_system    projections, folded into the cohorts
    GenTruth truth;
    OwnerSplit split;
    // Only the synthetic path can build the plaintext oracle, because the oracle
    // needs the UNION -- which is exactly what no party holds in a real run.
    const bool have_oracle = data_dir.empty();

    if (data_dir.empty()) {
        PlainBaseTable base = GenerateBaseTable(static_cast<size_t>(num_subjects), truth);
        split = SplitAcrossOwners(base, concentration, id_conflict_rate);
    } else {
        if (rows_a <= 0 || rows_b <= 0) {
            if (pID == 0)
                std::cerr << "FATAL: --data-dir needs --rows-a/--rows-b (the manifest "
                             "row counts, after the pass-1 visit_type filter). Both "
                             "halves are padded to one public length, and a party that "
                             "does not own a half cannot read its length from a file it "
                             "is not allowed to see."
                          << std::endl;
            return 1;
        }
        // With a single party in the protocol this process stands in for every
        // owner, so it must open both halves; otherwise each party opens only
        // its own and the other stays empty until the shares arrive.
        const bool simulate_all = (engine.getNumParties() <= 1);
        split.a = LoadOwnedBaseTable(data_dir, file_a, party_a, pID, simulate_all);
        split.b = LoadOwnedBaseTable(data_dir, file_b, party_b, pID, simulate_all);
    }

    // --- each owner, over its own rows, in the clear -------------------------
    // pcc_mrn and flagged_dx pass 1. Row-local, so this is the owner's own
    // plaintext and nothing here crosses the wire.
    ApplyMrnRepair(split.a);
    ApplyMrnRepair(split.b);
    PlainFlagged flagged_a = DeriveFlagged(split.a);
    PlainFlagged flagged_b = DeriveFlagged(split.b);

    size_t n_a = flagged_a.rows(), n_b = flagged_b.rows();
    if (!data_dir.empty()) {
        // A row-count disagreement does not fail cleanly later: the parties would
        // pad to different lengths and the shares would not line up.
        const bool simulate_all = (engine.getNumParties() <= 1);
        auto check = [&](int owner, size_t got, int declared, const char* which) {
            if ((simulate_all || pID == owner) && got != static_cast<size_t>(declared)) {
                std::cerr << "FATAL: owner " << which << "'s half has " << got
                          << " rows after the pass-1 filter but the manifest declares "
                          << declared << ". Every party pads from the manifest, so these "
                             "must agree exactly."
                          << std::endl;
                std::exit(1);
            }
        };
        check(party_a, flagged_a.rows(), rows_a, "A");
        check(party_b, flagged_b.rows(), rows_b, "B");
        n_a = static_cast<size_t>(rows_a);
        n_b = static_cast<size_t>(rows_b);
    }

    if (pID == 0 && !out_dir.empty() && have_oracle) {
        // Dump the two halves as the owners would hold them, so -O then -D is a
        // round trip.
        WriteBaseTableCsv(split.a, out_dir + "/base_owner_a.csv");
        WriteBaseTableCsv(split.b, out_dir + "/base_owner_b.csv");
        WriteFlaggedCsv(flagged_a, out_dir + "/flagged_dx_owner_a.csv");
        WriteFlaggedCsv(flagged_b, out_dir + "/flagged_dx_owner_b.csv");
        std::cout << "wrote both owners' halves of cdrcatsse_match_pcc to " << out_dir
                  << "\n  manifest row counts: --rows-a " << flagged_a.rows()
                  << " --rows-b " << flagged_b.rows() << std::endl;
    }

    // --- conflict_list, under MPC -------------------------------------------
    if (check_conflicts) {
        std::vector<DataType> ma, sa_, mb, sb_;
        for (size_t i = 0; i < split.a.rows(); ++i) {
            ma.push_back(MrnToInt(split.a.pat_mrn[i]));
            sa_.push_back(split.a.subject_id[i]);
        }
        for (size_t i = 0; i < split.b.rows(); ++i) {
            mb.push_back(MrnToInt(split.b.pat_mrn[i]));
            sb_.push_back(split.b.subject_id[i]);
        }
        const long secure_n =
            SecureConflictCount(engine, ma, sa_, party_a, mb, sb_, party_b);
        if (pID == 0 && !have_oracle) {
            std::cout << "\n=== conflict_list ===\n"
                      << "  MRNs mapping to more than one subject_id: " << secure_n
                      << "  (no plaintext cross-check: it would need both owners' rows)"
                      << std::endl;
        }
        if (pID == 0 && have_oracle) {
            // The oracle has to be the UNION OF WHAT THE OWNERS HOLD, after the
            // repair -- not the pre-split base table. A study-ID disagreement
            // between the two owners does not exist until the split creates it,
            // which is the whole reason this node cannot run at one owner.
            PlainBaseTable union_held;
            auto append = [&](const PlainBaseTable& t) {
                for (size_t i = 0; i < t.rows(); ++i)
                    union_held.PushRow(t.pat_mrn[i], t.subject_id[i], t.visit_type_pcc[i],
                                       t.visit_major_type_pccc[i], t.admit_date_pcc[i],
                                       t.diagnoses_pcc[i], t.visit_facility_pcc[i], t.dob[i],
                                       t.gender_pcc[i], t.hispanic[i]);
            };
            append(split.a);
            append(split.b);
            const long plain_n = static_cast<long>(ConflictList(union_held).size());
            std::cout << "\n=== conflict_list ===\n"
                      << "  MRNs mapping to more than one subject_id: " << secure_n
                      << "  (plaintext oracle over the union: " << plain_n << ")"
                      << (secure_n == plain_n ? "  MATCH" : "  *** MISMATCH ***")
                      << std::endl;
        }
    }

    // The plaintext mirror of the cross-party half, over the union. NOTHING in a
    // real deployment can compute this: it needs both halves, which is exactly
    // what no party holds. It exists only in the synthetic path, where every
    // party derives the same table from the same seed.
    PlainCohort any_plain, umass_plain, nonumass_plain;
    if (have_oracle) {
        any_plain = SequencePlain(flagged_a, flagged_b, SystemScope::Any);
        umass_plain = SequencePlain(flagged_a, flagged_b, SystemScope::UMass);
        nonumass_plain = SequencePlain(flagged_a, flagged_b, SystemScope::NonUMass);
    }

    // --- merge, sequence, re-sequence per system -----------------------------
    SecurePipeline pipeline =
        RunSecurePipeline(engine, flagged_a, party_a, flagged_b, party_b, n_a, n_b);
    SecureCohort& any = pipeline.any;
    SecureCohort& umass = pipeline.umass;
    SecureCohort& nonumass = pipeline.nonumass;

    // --- the relational stage against SQL, at the production row count -------
    //
    // Same three-way comparison -S kernels runs, but on this run's data. Needs
    // the union of both halves, so it is gated on have_oracle exactly as
    // SequencePlain is -- and the skip is printed rather than silent, because a
    // verification layer that quietly stops verifying is worse than none.
    if (check_sql) {
        // Collective: every party must open, whatever it will do with the result.
        const PlainCohort sql_mpc[] = {CohortFromSecure(any, pID),
                                       CohortFromSecure(umass, pID),
                                       CohortFromSecure(nonumass, pID)};
        if (pID == 0) {
            std::cout << "\n=== relational stage vs SQL ===" << std::endl;
#if defined(HAVE_SQLITE3)
            if (!have_oracle) {
                std::cout << "  SKIPPED: no plaintext oracle on the CSV path. The SQL\n"
                             "  oracle needs the union of both owners' halves, which is\n"
                             "  precisely what no party holds in a real deployment."
                          << std::endl;
            } else {
                SqlOracle sql(flagged_a, flagged_b);
                const PlainCohort* plain[] = {&any_plain, &umass_plain, &nonumass_plain};
                const SystemScope scopes[] = {SystemScope::Any, SystemScope::UMass,
                                              SystemScope::NonUMass};
                bool sql_ok = sql.ok();
                for (int i = 0; i < 3 && sql.ok(); ++i) {
                    const PlainCohort q = sql.Sequence(scopes[i]);
                    const SeqDiff d_plain = CompareSequencing(q, *plain[i]);
                    const SeqDiff d_mpc = CompareSequencing(q, sql_mpc[i]);
                    std::cout << "  " << ScopeName(scopes[i]) << std::endl;
                    ReportSeqDiff("  SQL vs SequencePlain", d_plain);
                    ReportSeqDiff("  SQL vs MPC", d_mpc);
                    sql_ok &= d_plain.ok() && d_mpc.ok();
                }
                std::cout << (sql_ok ? "  SQL CROSS-CHECK: PASS"
                                     : "  SQL CROSS-CHECK: *** FAIL ***")
                          << std::endl;
            }
#else
            std::cout << "  SKIPPED: built without SQLite (HAVE_SQLITE3 undefined)."
                      << std::endl;
#endif
        }
    }

    if (pID == 0) {
        std::cout << "\n################ MPC analysis pipeline ################\n"
                  << "source: " << (data_dir.empty() ? "synthetic base table" : data_dir)
                  << "\n"
                  << "owner A rows " << n_a << " (party " << party_a << "), "
                  << "owner B rows " << n_b << " (party " << party_b << ")\n"
                  << std::left << std::setw(20) << "table" << std::setw(12) << "rows"
                  << std::setw(12) << "padded" << std::setw(12) << "patients" << std::endl;
        for (const SecureCohort* c : {&any, &umass, &nonumass})
            std::cout << std::left << std::setw(20) << ScopeName(c->scope) << std::setw(12)
                      << c->n << std::setw(12) << c->n_pad << std::setw(12) << c->num_subjects
                      << std::endl;
        if (data_dir.empty())
            std::cout << "\ngenerating parameters: beta0=" << truth.beta0
                      << "  beta_visit=" << truth.beta_visit << "  beta_fu=" << truth.beta_fu
                      << "  beta_age=" << truth.beta_age << "  sigma=" << truth.sigma
                      << "  slope_diff(UMass-nonUMass)=" << truth.slope_diff << std::endl;
    }

    if (run_bench) {
        BenchmarkObjective(engine, pID, any);
        return 0;
    }

    // ------------------------------------------------------------- the counts
    if (print_describe) {
        ReportSisaCounts(any, pID);
        ReportSisaCounts(umass, pID);
        ReportSisaCounts(nonumass, pID);
    }

    if (!run_models) return 0;

    // ------------------------------------------------------------- the models
    const std::vector<ModelSpec> specs = AllModelSpecs();
    if (pID == 0)
        std::cout << "\n################ " << specs.size()
                  << " regression models ################" << std::endl;

    std::vector<FitResult> fits;
    for (const ModelSpec& spec : specs) {
        const SecureCohort& c = PickCohort(any, umass, nonumass, spec.scope);
        ModelData md = BuildDesign(c, spec);

        FitResult r = spec.random_intercept ? FitGlmmLaplace(md, spec, pID)
                                            : FitLogisticIrls(md, spec);
        fits.push_back(r);
        if (pID == 0) PrintFit(r, spec);

        // Score against the plaintext oracle. For the fixed-effects models this
        // is the same estimator in double precision, so the two should agree to
        // the fixed-point floor. For the mixed models it is NOT the same
        // estimand -- it ignores the random intercept -- so it is reported as
        // context, with the generating parameters as the real reference.
        // The oracle needs both halves, so it only exists in the synthetic path.
        if (pID == 0 && have_oracle) {
            const PlainCohort& pc = PickPlain(any_plain, umass_plain, nonumass_plain, spec.scope);
            std::vector<double> x_rm, y, mask;
            size_t pp = 0;
            PlainDesign(pc, spec, x_rm, y, mask, pp);
            PlainFit pf = PlainLogisticIrls(x_rm, y, mask, pc.rows(), pp, kIrlsRidge,
                                            kIrlsIterations);
            std::cout << "\n  " << (spec.random_intercept
                                        ? "plaintext IRLS (no random effect -- context only):"
                                        : "plaintext IRLS oracle (same estimator):")
                      << "\n  " << std::left << std::setw(34) << "Term" << std::setw(14)
                      << "MPC" << std::setw(14) << "Plaintext" << std::setw(12) << "Diff"
                      << std::endl;
            double worst = 0.0;
            for (size_t k = 0; k < r.terms.size() && k < pf.estimate.size(); ++k) {
                const double d = std::abs(r.estimate[k] - pf.estimate[k]);
                if (!spec.random_intercept) worst = std::max(worst, d);
                std::cout << "  " << std::left << std::setw(34) << r.terms[k] << std::fixed
                          << std::setprecision(5) << std::setw(14) << r.estimate[k]
                          << std::setw(14) << pf.estimate[k] << std::setw(12) << d
                          << std::defaultfloat << std::endl;
            }
            if (!spec.random_intercept)
                std::cout << "  max |MPC - plaintext| = " << std::scientific << worst
                          << std::defaultfloat
                          << (worst < 5e-3 ? "   OK" : "   *** CHECK ***") << std::endl;
        }
    }

    // --------------------------------------------------------------- summary
    if (pID == 0) {
        std::cout << "\n################ summary ################\n"
                  << std::left << std::setw(8) << "model" << std::setw(18) << "population"
                  << std::setw(32) << "time term" << std::setw(14) << "estimate"
                  << std::setw(12) << "SE" << std::setw(11) << "Pr>|z|" << std::endl;
        for (size_t i = 0; i < fits.size(); ++i) {
            const FitResult& r = fits[i];
            // The time coefficient is always the second term; for the
            // interaction models the estimand is the interaction, which is last.
            const size_t idx = specs[i].interaction ? r.terms.size() - 1 : 1;
            const bool have_se = std::isfinite(r.se[idx]) && r.se[idx] > 0.0;
            const double z = have_se ? r.estimate[idx] / r.se[idx] : 0.0;
            std::ostringstream se_s, p_s;
            if (have_se) {
                se_s << std::fixed << std::setprecision(5) << r.se[idx];
                p_s << FormatP(TwoSidedNormalP(z));
            } else {
                se_s << "n/a";
                p_s << "n/a";
            }
            std::cout << std::left << std::setw(8) << r.step << std::setw(18)
                      << ScopeLabel(r.scope) << std::setw(32) << r.terms[idx] << std::fixed
                      << std::setprecision(5) << std::setw(14) << r.estimate[idx]
                      << std::defaultfloat << std::setw(12) << se_s.str() << std::setw(11)
                      << p_s.str() << std::endl;
        }
        if (data_dir.empty())
            std::cout << "\ngenerating truth: visit_num slope " << truth.beta_visit
                      << ", fu_month slope " << truth.beta_fu << ", UMass slope difference "
                      << truth.slope_diff << " (models 6a/6b)" << std::endl;
        std::cout << "\nThe two step-6 models are the comparison the script is built to make:\n"
                     "their interaction coefficient IS the difference in time slopes between\n"
                     "the UMass and non-UMass systems. The paired subset models cannot answer\n"
                     "it -- data_source does not vary inside either subset."
                  << std::endl;

        // Machine-readable form, for scripts/testing/validate_mpc_analysis.py
        // --compare. One tab-separated line per coefficient, plus one per
        // variance component. Kept deliberately dull so it stays parseable.
        std::cout << "\n# RESULT step\tpopulation\tterm\testimate\tse" << std::endl;
        for (size_t i = 0; i < fits.size(); ++i) {
            const FitResult& fit = fits[i];
            const char* pop = fit.scope == SystemScope::Any
                                  ? "any"
                                  : (fit.scope == SystemScope::UMass ? "umass" : "nonumass");
            for (size_t k = 0; k < fit.terms.size(); ++k) {
                std::cout << "RESULT\t" << fit.step << '\t' << pop << '\t' << fit.terms[k]
                          << '\t' << std::setprecision(9) << fit.estimate[k] << '\t';
                if (std::isfinite(fit.se[k]))
                    std::cout << fit.se[k];
                else
                    std::cout << "nan";
                std::cout << std::endl;
            }
            if (fit.has_random)
                std::cout << "RESULT\t" << fit.step << '\t' << pop
                          << "\trandom_intercept_variance\t" << std::setprecision(9)
                          << fit.sigma2 << "\tnan" << std::endl;
        }
    }
    return 0;
}

// MPC port of the SISA acute-care analysis pipeline.
//   https://cs-people.bu.edu/liagos/pilot/mpc_analysis_lineage.html
//
// Three patient-encounter tables in (any_system, umass_system, nonumass_system);
// nineteen terminal outputs out -- two printed descriptive reports, three
// one-row aggregate tables, and fourteen fitted regression models (six
// specifications across three populations, minus the four interaction fits that
// can only run on the pooled table). Nothing downstream consumes anything else:
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
//   nodes.h        d1a, d1b and sisa_perct_cnt x3
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
//   ./mpc-analysis -S describe -r 200         # ingestion + descriptive nodes
//   ./mpc-analysis -S models -r 200           # the fourteen fits
//   ./mpc-analysis -S bench -r 200            # cost breakdown of one fit
//   ./mpc-analysis -S models -r 200 -C 0      # ... with uncached segmented scans
//   ./mpc-analysis -D /data                   # read per-party CSVs instead
//   ./mpc-analysis -O /tmp/dump               # dump the synthetic cohort to CSV

#include "cdough.h"

#include "./harness.h"
#include "./reporting.h"

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
    //   describe - ingestion plus the descriptive and aggregate nodes
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
    // Directory holding any_system.csv / umass_system.csv / nonumass_system.csv.
    // Empty means "use the synthetic generator".
    const std::string data_dir = engine.getArg<std::string>("data-dir", "D", "");
    // Where to dump the synthetic cohort for the R / Python / SQLite cross-checks.
    const std::string out_dir = engine.getArg<std::string>("out-dir", "O", "");
    // Which party holds each table in the clear.
    const int party_any = engine.getArg<int>("party-any", "pa", 0);
    const int party_umass = engine.getArg<int>("party-umass", "pu", 0);
    const int party_nonumass = engine.getArg<int>("party-nonumass", "pn", 0);
    // 0 routes the segmented scans through aggregators::aggregate instead of the
    // cached per-level group bits, for A/B measurement.
    g_use_cached_scans = engine.getArg<int>("cached-scans", "C", 1) != 0;

    if (run_kernels) {
        TestNewKernels(engine, pID);
        TestSegmented(engine, pID);
        TestLinearAlgebra(engine, pID);
    }
    if (!run_describe && !run_bench) return 0;

    // ---------------------------------------------------------------- ingest
    SyntheticTruth truth;
    PlainCohort any_plain, umass_plain, nonumass_plain;

    if (data_dir.empty()) {
        any_plain = MakeSyntheticCohort(static_cast<size_t>(num_subjects), truth);
        umass_plain = DeriveSystemCohort(any_plain, SystemScope::UMass);
        nonumass_plain = DeriveSystemCohort(any_plain, SystemScope::NonUMass);
        if (pID == 0 && !out_dir.empty()) {
            WriteCohortCsv(any_plain, out_dir + "/" + CohortCsvName(SystemScope::Any));
            WriteCohortCsv(umass_plain, out_dir + "/" + CohortCsvName(SystemScope::UMass));
            WriteCohortCsv(nonumass_plain,
                           out_dir + "/" + CohortCsvName(SystemScope::NonUMass));
            std::cout << "wrote the synthetic cohort to " << out_dir << std::endl;
        }
    } else {
        any_plain = ReadCohortCsv(data_dir + "/" + CohortCsvName(SystemScope::Any),
                                  SystemScope::Any);
        umass_plain = ReadCohortCsv(data_dir + "/" + CohortCsvName(SystemScope::UMass),
                                    SystemScope::UMass);
        nonumass_plain = ReadCohortCsv(data_dir + "/" + CohortCsvName(SystemScope::NonUMass),
                                       SystemScope::NonUMass);
    }

    SecureCohort any = ShareCohort(engine, any_plain, party_any);
    SecureCohort umass = ShareCohort(engine, umass_plain, party_umass);
    SecureCohort nonumass = ShareCohort(engine, nonumass_plain, party_nonumass);

    if (pID == 0) {
        std::cout << "\n################ MPC analysis pipeline ################\n"
                  << "source: " << (data_dir.empty() ? "synthetic cohort" : data_dir) << "\n"
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

    // ------------------------------------------------- descriptive and counts
    if (print_describe) {
        long max_visits = 1;
        for (size_t i = 0; i < any_plain.rows(); ++i)
            max_visits = std::max<long>(max_visits, static_cast<long>(any_plain.visit_num[i]));
        // A public upper bound for the histogram sweep. The counts themselves are
        // the published output of the node.
        ReportD1a(any, pID, max_visits);
        ReportD1b(any, pID);
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
        if (pID == 0) {
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

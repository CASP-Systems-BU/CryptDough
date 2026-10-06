// MPC port of the SISA acute-care analysis pipeline.
//   https://cs-people.bu.edu/liagos/pilot/mpc_analysis_lineage.html
//
// Two owners' halves of cdrcatsse_match_pcc in -- or ONE owner's whole table,
// with --owner -- and seventeen terminal outputs out --
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
//                  SecureReciprocal, SecureSqrt, Exp/Log/Log1p,
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
//     appear on a hot path; SecureReciprocal is the one division operator.
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
//   ./mpc-analysis -S models -r 200 -neng 4  # the fourteen fits spread over 4
//                                             # engines, one thread each (semantic
//                                             # task 0024; replicated 3PC only)
//
// Running ONE node, for ONE data owner (see tasks/0013):
//
//   ./mpc-analysis -N 5a:umass -ow 0 \
//       -D /data -F input.csv -rs 1310 -o /out/5a_umass.csv
//
//     -N  which lineage node to run; -N all (default) runs the whole pipeline.
//         Relational query steps, aggregate nodes and individual model fits are
//         all addressable -- see `-N ?` for the list.
//     -ow the SINGLE data owner. That party holds the one input CSV, secret-
//         shares it to the others, and is the only party the node's output is
//         opened to. Omit it for the two-owner data model, where outputs are
//         published to everybody.
//     -F  the owner's CSV, inside -D. Only party -ow opens it.
//     -rs its row count AFTER the pass-1 visit_type filter, from the manifest --
//         the single-owner counterpart of -ra/-rb, and public for the same
//         reason: every party pads from it and the non-owners cannot read the
//         file's length.
//     -o  where the owner writes the node's result, as CSV.

#include <algorithm>
#include <chrono>
#include <numeric>
#include <thread>

#include "cdough.h"

#include "./harness.h"
#include "./output.h"
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

// ReportSisaCounts caches its scan plan, so it needs a mutable cohort.
SecureCohort& PickCohortMut(SecureCohort& any, SecureCohort& umass, SecureCohort& nonumass,
                            SystemScope s) {
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

// =============================================================================
// Parallel model fits (semantic task 0024)
//
// With -neng K > 1 the fourteen fits are spread over K independent engines, one
// thread each. Two rules make that correct rather than correct by luck:
//
//   1. The engines are created one at a time on the main thread. Communicators
//      are matched across parties by MPI tag, and tags are handed out in
//      creation order, so engine k at party 0 pairs with engine k at every other
//      party only if every party creates them in the same order.
//   2. Which engine fits which model is a fixed function of the spec list and
//      K, never a work queue. A queue would let the parties pick different fits
//      on the same engine, and the run would hang or compute garbage.
//
// Each engine works on its own copy of the three cohorts (CloneCohortOnto);
// nothing is shared between threads except read-only plaintext.
// =============================================================================

// The three analysis tables, bound to one engine.
struct CohortSet {
    SecureCohort any;
    SecureCohort umass;
    SecureCohort nonumass;
};

// Relative cost of one fit, as a static weight for the schedule. Measured on the
// synthetic -r 200 cohort (semantic task 0024 pilot): a BFGS iteration costs
// ~13.5 s whatever the population, because all three tables share one padded
// length, so a mixed fit's cost is its iteration count. The covariate models
// (5a/5b) ran to the 25-iteration cap (~340 s); the unadjusted ones took 9-19
// (~115-250 s); IRLS is a handful of Cholesky solves. Only the ORDER these
// weights induce matters, and the schedule must not depend on anything a party
// learns at run time.
int FitCostWeight(const ModelSpec& spec) {
    if (!spec.random_intercept) return 1;
    return spec.covars ? 6 : 3;
}

// Longest-processing-time-first, by the static weights above: take the fits
// heaviest first and give each to the engine with the least weight so far,
// lowest index on ties. Deterministic, so every party computes the same plan.
// With one engine this is the spec order unchanged.
std::vector<std::vector<size_t>> ScheduleFits(const std::vector<ModelSpec>& specs,
                                              size_t num_engines) {
    std::vector<std::vector<size_t>> plan(num_engines);
    if (num_engines == 1) {
        for (size_t i = 0; i < specs.size(); ++i) plan[0].push_back(i);
        return plan;
    }
    std::vector<size_t> order(specs.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return FitCostWeight(specs[a]) > FitCostWeight(specs[b]);
    });
    std::vector<int> load(num_engines, 0);
    for (size_t i : order) {
        const size_t k = static_cast<size_t>(
            std::min_element(load.begin(), load.end()) - load.begin());
        plan[k].push_back(i);
        load[k] += FitCostWeight(specs[i]);
    }
    return plan;
}

FitResult FitOne(const SecureCohort& c, const ModelSpec& spec, int reveal_to, int party_id) {
    ModelData md = BuildDesign(c, spec, reveal_to, party_id);
    return spec.random_intercept ? FitGlmmLaplace(md, spec, party_id, reveal_to)
                                 : FitLogisticIrls(md, spec, reveal_to, party_id);
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
    bool run_kernels = (stage == "kernels" || stage == "all");
    bool run_describe = (stage == "describe" || stage == "models" || stage == "all");
    bool run_models = (stage == "models" || stage == "all");
    bool run_bench = (stage == "bench");
    bool print_describe = (stage == "describe" || stage == "all");

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
    // Engines to spread the model fits over (semantic task 0024). The runtime
    // itself registers this flag; 1 keeps the original sequential loop.
    int num_engines = engine.getArg<int>("num-engines", "neng", 1);
    if (num_engines < 1) {
        if (pID == 0) std::cerr << "FATAL: -neng must be at least 1." << std::endl;
        return 1;
    }
#if !(PROTOCOL_NUM == REPLICATED3 && \
      (defined(MPC_USE_MPI_COMMUNICATOR) || defined(MPC_USE_NO_COPY_COMMUNICATOR)))
    // CloneCohortOnto is only valid where a share depends on nothing but the
    // party ID, and the engine pairing is verified for the MPI and no-copy
    // communicators only (semantic tasks 0024, 0025).
    if (num_engines > 1) {
        if (pID == 0)
            std::cerr << "FATAL: -neng > 1 is validated only for replicated 3PC over the MPI "
                         "or no-copy communicator (semantic tasks 0024, 0025)."
                      << std::endl;
        return 1;
    }
#endif

    // ---------------------------------------------- one node, one data owner
    //
    // Which lineage node to run. `all` keeps the whole-pipeline behaviour; any
    // other value computes that node and nothing else.
    const std::string node_arg = engine.getArg<std::string>("node", "N", "all");
    // The SINGLE data owner, when there is one: the party that holds the one
    // input CSV, secret-shares it to the others, and receives the output. -1
    // selects the two-owner data model, where -pa / -pb each hold a half and
    // every output is published to all parties.
    const int owner = engine.getArg<int>("owner", "ow", -1);
    const std::string file_one = engine.getArg<std::string>("file", "F", "base_owner.csv");
    // Row count after the pass-1 visit_type filter -- the single-owner
    // counterpart of -ra / -rb, public for the same reason.
    const int rows_one = engine.getArg<int>("rows", "rs", 0);
    // Where the owner writes the node's result.
    const std::string out_file = engine.getArg<std::string>("out-file", "o", "");

    const bool single_owner = (owner >= 0);
    // -1 is the framework's own open(): every party learns the value. With a
    // single owner the output belongs to that party alone, and every open below
    // is masked so that it does.
    const int reveal_to = single_owner ? owner : -1;

    if (node_arg == "?" || node_arg == "help" || node_arg == "list") {
        if (pID == 0) std::cout << "-N accepts:\n" << kNodeHelp << std::endl;
        return 0;
    }
    NodeRequest node;
    {
        std::string err;
        if (!ParseNode(node_arg, node, err)) {
            if (pID == 0)
                std::cerr << "FATAL: -N " << node_arg << ": " << err << "\n\n-N accepts:\n"
                          << kNodeHelp << std::endl;
            return 1;
        }
    }
    const bool one_node = (node.kind != NodeKind::All);

    if (single_owner && owner >= engine.getNumParties()) {
        if (pID == 0)
            std::cerr << "FATAL: --owner " << owner << " is not a party in this run ("
                      << engine.getNumParties() << " parties)." << std::endl;
        return 1;
    }
    if (single_owner && check_sql) {
        if (pID == 0)
            std::cerr << "FATAL: -Q opens all three analysis tables to EVERY party so they "
                         "can be diffed against SQLite, which is the opposite of what "
                         "--owner asks for. Run the cross-check without --owner."
                      << std::endl;
        return 1;
    }
    if (!out_file.empty() && !one_node) {
        if (pID == 0)
            std::cerr << "FATAL: -o writes ONE node's result, so it needs -N to say which."
                      << std::endl;
        return 1;
    }

    // Naming a node is its own stage: it needs ingestion whatever -S says, and
    // it runs nothing else. The accuracy harnesses are minutes of work that say
    // nothing about the node, so they run only when -S kernels asks for them by
    // name rather than falling out of the default -S all.
    if (one_node) {
        run_kernels = (stage == "kernels");
        run_describe = true;
        run_models = false;
        run_bench = false;
        print_describe = false;
    }

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
    // (With a single owner that party DOES hold the union, but the other parties
    // do not, so building it here would make them diverge; the oracle stays off.)
    const bool have_oracle = data_dir.empty();

    // With one data owner everything lives in half A and half B stays empty, so
    // the ingestion, the MRN repair and pass 1 below are shared with the
    // two-owner path unchanged. Only the shape of the secure table differs, and
    // that difference is confined to RunSecurePipelineSingleOwner.
    const int in_party_a = single_owner ? owner : party_a;
    const int in_party_b = single_owner ? owner : party_b;

    // With a single party in the protocol this process stands in for every
    // owner, so it must open every file; otherwise each party opens only its
    // own and the rest stay empty until the shares arrive.
    const bool simulate_all = (engine.getNumParties() <= 1);

    if (data_dir.empty()) {
        PlainBaseTable base = GenerateBaseTable(static_cast<size_t>(num_subjects), truth);
        if (single_owner)
            split.a = base;  // one owner holds the whole of cdrcatsse_match_pcc
        else
            split = SplitAcrossOwners(base, concentration, id_conflict_rate);
    } else if (single_owner) {
        if (rows_one <= 0) {
            if (pID == 0)
                std::cerr << "FATAL: --owner with --data-dir needs --rows (the manifest "
                             "row count, after the pass-1 visit_type filter). The table "
                             "is padded to one public length, and the parties that do "
                             "not own the file cannot read its length."
                          << std::endl;
            return 1;
        }
        split.a = LoadOwnedBaseTable(data_dir, file_one, owner, pID, simulate_all);
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
        auto check = [&](int holder, size_t got, int declared, const char* which) {
            if ((simulate_all || pID == holder) && got != static_cast<size_t>(declared)) {
                std::cerr << "FATAL: owner " << which << "'s half has " << got
                          << " rows after the pass-1 filter but the manifest declares "
                          << declared << ". Every party pads from the manifest, so these "
                             "must agree exactly."
                          << std::endl;
                std::exit(1);
            }
        };
        if (single_owner) {
            check(owner, flagged_a.rows(), rows_one, "");
            n_a = static_cast<size_t>(rows_one);
            n_b = 0;
        } else {
            check(party_a, flagged_a.rows(), rows_a, "A");
            check(party_b, flagged_b.rows(), rows_b, "B");
            n_a = static_cast<size_t>(rows_a);
            n_b = static_cast<size_t>(rows_b);
        }
    }

    if (pID == 0 && !out_dir.empty() && have_oracle) {  // simulation-only dump
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
    //
    // -N conflict_list asks for this node alone, and it does not need the merge
    // or either sequencing pass, so the run stops here rather than paying for
    // them. With a single owner the node is degenerate -- one organisation can
    // see its own MRN collisions in the clear -- and it is answered anyway, on
    // the owner's rows, so the two data models take the same code path.
    long conflict_count = -1;
    const bool want_conflicts = check_conflicts || node.kind == NodeKind::ConflictList;
    const int conflict_recipient = reveal_to < 0 ? 0 : reveal_to;
    if (want_conflicts) {
        std::vector<DataType> ma, sa_, mb, sb_;
        for (size_t i = 0; i < split.a.rows(); ++i) {
            ma.push_back(MrnToInt(split.a.pat_mrn[i]));
            sa_.push_back(split.a.subject_id[i]);
        }
        for (size_t i = 0; i < split.b.rows(); ++i) {
            mb.push_back(MrnToInt(split.b.pat_mrn[i]));
            sb_.push_back(split.b.subject_id[i]);
        }
        const long secure_n = SecureConflictCount(engine, ma, sa_, in_party_a, mb, sb_,
                                                 in_party_b, reveal_to, pID);
        conflict_count = secure_n;
        if (pID == conflict_recipient && !have_oracle) {
            std::cout << "\n=== conflict_list ===\n"
                      << "  MRNs mapping to more than one subject_id: " << secure_n
                      << "  (no plaintext cross-check: it would need both owners' rows)"
                      << std::endl;
        }
        if (pID == conflict_recipient && have_oracle) {
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

    if (node.kind == NodeKind::ConflictList) {
        if (pID == conflict_recipient && !out_file.empty()) {
            if (!WriteConflictCsv(out_file, conflict_count)) return 1;
            std::cout << "\nwrote conflict_list to " << out_file << std::endl;
        }
        return 0;
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
    //
    // One owner needs no merge: it sorted its whole table locally, so the shared
    // table is already ordered and the log N compare-swap stages have nothing to
    // do. It also pads to NextPowerOfTwo(n) rather than twice that, because
    // there is no second half to match. Everything after ingestion is identical,
    // which is why both calls land in the same SequenceSharedTable.
    SecurePipeline pipeline =
        single_owner
            ? RunSecurePipelineSingleOwner(engine, flagged_a, owner, n_a)
            : RunSecurePipeline(engine, flagged_a, party_a, flagged_b, party_b, n_a, n_b);
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

    const int banner_party = reveal_to < 0 ? 0 : reveal_to;
    if (pID == banner_party) {
        std::ostringstream owners;
        if (single_owner)
            owners << "sole owner: party " << owner << ", " << n_a << " rows (file "
                   << (data_dir.empty() ? "synthetic" : file_one)
                   << "); output opened to that party only";
        else
            owners << "owner A rows " << n_a << " (party " << party_a << "), "
                   << "owner B rows " << n_b << " (party " << party_b << ")";
        std::cout << "\n################ MPC analysis pipeline ################\n"
                  << "source: " << (data_dir.empty() ? "synthetic base table" : data_dir)
                  << "\n"
                  << owners.str() << "\n"
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

    // ----------------------------------------------------- one node, and stop
    //
    // Everything above is shared with the full run: ingestion and the relational
    // stage are what every node reads. What differs is that exactly one terminal
    // output is computed, and that with --owner its opens are masked so only the
    // owner sees it -- the other parties reach every open, get nothing back, and
    // write nothing.
    if (one_node) {
        const bool mine = (pID == (reveal_to < 0 ? 0 : reveal_to));
        switch (node.kind) {
            case NodeKind::Relational: {
                const SecureCohort& c = PickCohort(any, umass, nonumass, node.scope);
                const PlainCohort table = OpenCohortToParty(c, reveal_to, pID);
                if (mine) {
                    std::cout << "\n=== " << ScopeName(node.scope) << "  ["
                              << ScopeLabel(node.scope) << "] ===\n"
                              << "  " << table.rows() << " encounters, " << c.num_subjects
                              << " patients" << std::endl;
                    if (!out_file.empty() && !WriteCohortCsv(out_file, table)) return 1;
                }
                break;
            }
            case NodeKind::Counts: {
                SecureCohort& c = PickCohortMut(any, umass, nonumass, node.scope);
                const std::vector<SisaCounts> rows = ReportSisaCounts(c, pID, reveal_to);
                if (mine && !out_file.empty() && !WriteCountsCsv(out_file, node.scope, rows))
                    return 1;
                break;
            }
            case NodeKind::Model: {
                // DEBUG ONLY (semantic task 0024, experiment 0003 post-hoc): fit the same
                // node on -neng engines, to separate "running concurrently" from "running
                // on a worker thread". `-dbg-fit serial` runs one worker thread per engine,
                // each joined before the next starts; `concurrent` starts them all at once.
                // The default, `main`, is the ordinary single-node path below.
                const std::string dbg_fit = engine.getArg<std::string>("dbg-fit", "dbgf", "main");
                if (dbg_fit != "main") {
                    // `mainextra`: create the extra engines, then fit engine 0 only, on THIS
                    // thread -- isolates "another engine exists" from "worker thread".
                    if (dbg_fit != "serial" && dbg_fit != "concurrent" && dbg_fit != "mainextra") {
                        if (pID == 0)
                            std::cerr << "FATAL: -dbg-fit must be main, serial, concurrent or "
                                         "mainextra"
                                      << std::endl;
                        return 1;
                    }
                    std::vector<Engine*> dbg_engines{&engine};
                    for (int k = 1; k < num_engines; ++k)
                        dbg_engines.push_back(&cdough_init(argc, argv));
                    std::vector<CohortSet> dbg_cohorts;
                    dbg_cohorts.reserve(dbg_engines.size());
                    for (size_t k = 1; k < dbg_engines.size(); ++k)
                        dbg_cohorts.push_back(CohortSet{CloneCohortOnto(any, *dbg_engines[k]),
                                                        CloneCohortOnto(umass, *dbg_engines[k]),
                                                        CloneCohortOnto(nonumass, *dbg_engines[k])});
                    std::vector<FitResult> dbg_fits(dbg_engines.size());
                    auto fit_on = [&](size_t k) {
                        const SecureCohort& c_k =
                            (k == 0) ? PickCohort(any, umass, nonumass, node.spec.scope)
                                     : PickCohort(dbg_cohorts[k - 1].any, dbg_cohorts[k - 1].umass,
                                                  dbg_cohorts[k - 1].nonumass, node.spec.scope);
                        dbg_fits[k] = FitOne(c_k, node.spec, reveal_to, pID);
                    };
                    if (dbg_fit == "mainextra") dbg_fits.resize(1);
                    std::vector<std::thread> dbg_threads;
                    if (dbg_fit == "mainextra") fit_on(0);
                    for (size_t k = 0; dbg_fit != "mainextra" && k < dbg_engines.size(); ++k) {
                        dbg_threads.emplace_back(fit_on, k);
                        if (dbg_fit == "serial") dbg_threads.back().join();
                    }
                    for (std::thread& t : dbg_threads)
                        if (t.joinable()) t.join();
                    if (mine) {
                        for (size_t k = 0; k < dbg_fits.size(); ++k) {
                            std::ostringstream line;
                            line << "[DBG] mode=" << dbg_fit << " engine=" << k
                                 << " iters=" << dbg_fits[k].iterations
                                 << " cond=" << dbg_fits[k].hessian_condition << std::setprecision(9);
                            for (size_t t = 0; t < dbg_fits[k].terms.size(); ++t)
                                line << "  " << dbg_fits[k].terms[t] << '=' << dbg_fits[k].estimate[t];
                            line << "  sigma2=" << dbg_fits[k].sigma2 << '\n';
                            std::cout << line.str() << std::flush;
                        }
                    }
                    break;
                }
                const SecureCohort& c = PickCohort(any, umass, nonumass, node.spec.scope);
                ModelData md = BuildDesign(c, node.spec, reveal_to, pID);
                const FitResult fit =
                    node.spec.random_intercept
                        ? FitGlmmLaplace(md, node.spec, pID, reveal_to)
                        : FitLogisticIrls(md, node.spec, reveal_to, pID);
                if (mine) {
                    PrintFit(fit, node.spec);
                    if (!out_file.empty() && !WriteFitCsv(out_file, fit, node.spec)) return 1;
                }
                break;
            }
            default:
                break;  // ConflictList returned above; All is not one_node
        }
        if (mine && !out_file.empty())
            std::cout << "\nwrote " << node.name << " to " << out_file << std::endl;
        return 0;
    }

    if (run_bench) {
        BenchmarkObjective(engine, pID, any);
        return 0;
    }

    // ------------------------------------------------------------- the counts
    if (print_describe) {
        ReportSisaCounts(any, pID, reveal_to);
        ReportSisaCounts(umass, pID, reveal_to);
        ReportSisaCounts(nonumass, pID, reveal_to);
    }

    if (!run_models) return 0;

    // ------------------------------------------------------------- the models
    const std::vector<ModelSpec> specs = AllModelSpecs();
    // More engines than fits would only add idle runtimes.
    num_engines = std::min<int>(num_engines, static_cast<int>(specs.size()));
    if (pID == banner_party)
        std::cout << "\n################ " << specs.size()
                  << " regression models ################" << std::endl;

    // Print one fit, and score it against the plaintext oracle. For the
    // fixed-effects models the oracle is the same estimator in double precision,
    // so the two should agree to the fixed-point floor. For the mixed models it
    // is NOT the same estimand -- it ignores the random intercept -- so it is
    // reported as context, with the generating parameters as the real reference.
    // The oracle needs both halves, so it only exists in the synthetic path.
    auto report_fit = [&](const FitResult& r, const ModelSpec& spec) {
        if (pID != banner_party) return;
        PrintFit(r, spec);
        if (!have_oracle) return;
        const PlainCohort& pc = PickPlain(any_plain, umass_plain, nonumass_plain, spec.scope);
        std::vector<double> x_rm, y, mask;
        size_t pp = 0;
        PlainDesign(pc, spec, x_rm, y, mask, pp);
        PlainFit pf =
            PlainLogisticIrls(x_rm, y, mask, pc.rows(), pp, kIrlsRidge, kIrlsIterations);
        std::cout << "\n  " << (spec.random_intercept
                                    ? "plaintext IRLS (no random effect -- context only):"
                                    : "plaintext IRLS oracle (same estimator):")
                  << "\n  " << std::left << std::setw(34) << "Term" << std::setw(14) << "MPC"
                  << std::setw(14) << "Plaintext" << std::setw(12) << "Diff" << std::endl;
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
                      << std::defaultfloat << (worst < 5e-3 ? "   OK" : "   *** CHECK ***")
                      << std::endl;
    };

    // Wall clock per fit and for the whole stage, for the speedup measurement
    // (semantic experiment 0003). Elapsed time is not derived from any share.
    using FitClock = std::chrono::steady_clock;
    auto fit_seconds = [](FitClock::time_point since) {
        return std::chrono::duration<double>(FitClock::now() - since).count();
    };
    auto log_fit_time = [&](size_t i, size_t k, double seconds) {
        if (pID != banner_party) return;
        std::ostringstream line;
        line << "[FIT] " << specs[i].step << ' ' << ScopeName(specs[i].scope) << "  engine "
             << k << "  time=" << std::fixed << std::setprecision(3) << seconds << "s\n";
        std::cout << line.str() << std::flush;
    };
    const FitClock::time_point models_start = FitClock::now();

    std::vector<FitResult> fits(specs.size());
    if (num_engines == 1) {
        // The original sequential loop: each fit is printed as soon as it ends.
        for (size_t i = 0; i < specs.size(); ++i) {
            const FitClock::time_point t0 = FitClock::now();
            fits[i] = FitOne(PickCohort(any, umass, nonumass, specs[i].scope), specs[i],
                             reveal_to, pID);
            log_fit_time(i, 0, fit_seconds(t0));
            report_fit(fits[i], specs[i]);
        }
    } else {
        // Extra engines, created one at a time on this thread -- see rule 1 above
        // ScheduleFits. Each creation is collective: every party runs it here, in
        // the same order.
        std::vector<Engine*> engines{&engine};
        for (int k = 1; k < num_engines; ++k) engines.push_back(&cdough_init(argc, argv));

        // Engine 0 keeps the cohorts it built; every other engine gets a copy.
        // Local copies only: no communication, so no ordering constraint.
        std::vector<CohortSet> cohorts;
        cohorts.reserve(engines.size() - 1);
        for (size_t k = 1; k < engines.size(); ++k)
            cohorts.push_back(CohortSet{CloneCohortOnto(any, *engines[k]),
                                        CloneCohortOnto(umass, *engines[k]),
                                        CloneCohortOnto(nonumass, *engines[k])});

        const std::vector<std::vector<size_t>> plan = ScheduleFits(specs, engines.size());
        if (pID == banner_party) {
            std::cout << "fitting on " << engines.size() << " engines in parallel" << std::endl;
            for (size_t k = 0; k < plan.size(); ++k) {
                std::cout << "  engine " << k << ":";
                for (size_t i : plan[k])
                    std::cout << ' ' << specs[i].step << '/' << ScopeName(specs[i].scope);
                std::cout << std::endl;
            }
        }

        // Each thread writes only fits[i] for the i it was assigned, so the
        // slots never overlap and the vector itself is never resized.
        std::vector<std::thread> workers;
        workers.reserve(engines.size());
        for (size_t k = 0; k < engines.size(); ++k) {
            workers.emplace_back([&, k]() {
                const SecureCohort& c_any = (k == 0) ? any : cohorts[k - 1].any;
                const SecureCohort& c_umass = (k == 0) ? umass : cohorts[k - 1].umass;
                const SecureCohort& c_nonumass = (k == 0) ? nonumass : cohorts[k - 1].nonumass;
                for (size_t i : plan[k]) {
                    const FitClock::time_point t0 = FitClock::now();
                    fits[i] = FitOne(PickCohort(c_any, c_umass, c_nonumass, specs[i].scope),
                                     specs[i], reveal_to, pID);
                    log_fit_time(i, k, fit_seconds(t0));
                }
            });
        }
        for (std::thread& t : workers) t.join();

        // Everything is reported after the join, in spec order, so the tables
        // and the summary below read exactly as a sequential run's do.
        for (size_t i = 0; i < specs.size(); ++i) report_fit(fits[i], specs[i]);
    }
    if (pID == banner_party)
        std::cout << "\n[FIT] models stage  engines " << num_engines << "  time=" << std::fixed
                  << std::setprecision(3) << fit_seconds(models_start) << "s"
                  << std::defaultfloat << std::endl;

    // --------------------------------------------------------------- summary
    if (pID == banner_party) {
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

#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "./cohort.h"
#include "./nodes.h"
#include "./primitives.h"
#include "./regression.h"
#include "./reporting.h"

// Running ONE lineage node, and handing its result to the party that owns it.
//
// The driver's default is still the whole pipeline: ingest once, then fan out to
// every terminal output. This header adds the other mode the lineage page
// implies -- name a single node, compute that node and nothing else, and write
// its result out in the form the node actually has.
//
// Three things live here:
//
//   NodeRequest / ParseNode   the `--node` grammar and what it selects
//   OpenCohortToParty         one relational query step's result, opened to its
//                             owner and nobody else
//   Write*Csv                 the result files, one shape per kind of node
//
// The file format follows the node, not the other way round. A relational query
// step is a TABLE, so it lands as one CSV row per encounter with the analysis
// table's own column names. An aggregate node is a small table, so it lands as
// one row per follow-up window. A fitted model is a coefficient table, so it
// lands as one row per term with the derived statistics (z, p, odds ratio and
// its interval) alongside -- the same columns PrintFit puts on the terminal, in
// a form something else can read. Every file is plain RFC 4180 CSV with a header
// line: no comment banner, because a comment banner stops it being a CSV.

namespace cdough::regression {

// =============================================================================
// The --node grammar
// =============================================================================

enum class NodeKind {
    All,           // the whole pipeline: every terminal output (the default)
    Relational,    // one of the three analysis tables
    Counts,        // one sisa_perct_cnt table
    Model,         // one (specification, population) fit
    ConflictList,  // the cross-party MRN check
};

struct NodeRequest {
    NodeKind kind = NodeKind::All;
    std::string name = "all";         // as the user wrote it, for labelling
    SystemScope scope = SystemScope::Any;
    ModelSpec spec;                   // meaningful only when kind == Model
};

// `umass` / `umass_system` / `UMass` all name the same population.
bool ParseScope(const std::string& raw, SystemScope& out) {
    std::string s;
    for (char c : raw) s += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == "any" || s == "any_system" || s == "all" || s == "pooled") {
        out = SystemScope::Any;
        return true;
    }
    if (s == "umass" || s == "umass_system") {
        out = SystemScope::UMass;
        return true;
    }
    if (s == "nonumass" || s == "nonumass_system" || s == "non-umass") {
        out = SystemScope::NonUMass;
        return true;
    }
    return false;
}

// The six specifications, as AllModelSpecs builds them, for one population.
// Kept here rather than in the driver so `--node 5a:umass` and the full run
// cannot disagree about what model 5a is.
bool ModelSpecFor(const std::string& step, SystemScope scope, ModelSpec& out,
                  std::string& error) {
    struct Base {
        const char* step;
        TimeAxis time;
        bool covars;
        bool interaction;
    };
    static const std::vector<Base> bases = {
        {"2a", TimeAxis::VisitNum, false, false}, {"2b", TimeAxis::FuMonth, false, false},
        {"5a", TimeAxis::VisitNum, true, false},  {"5b", TimeAxis::FuMonth, true, false},
        {"6a", TimeAxis::VisitNum, true, true},   {"6b", TimeAxis::FuMonth, true, true},
    };
    for (const Base& b : bases) {
        if (step != b.step) continue;
        if (b.interaction && scope != SystemScope::Any) {
            error = "model " + step +
                    " can only be fitted to any_system: data_source does not vary inside "
                    "either subset, which is the whole reason the script is built around "
                    "the pooled table";
            return false;
        }
        out.step = b.step;
        out.time = b.time;
        out.covars = b.covars;
        out.interaction = b.interaction;
        out.random_intercept = !b.interaction;
        out.scope = scope;
        // The SAS comments hispanic out of model 5b on the UMass subset, citing
        // sparse data.
        out.drop_hispanic = (out.step == "5b" && scope == SystemScope::UMass);
        return true;
    }
    error = "unknown model step '" + step + "'";
    return false;
}

const char* kNodeHelp =
    "  all                                  every terminal output (default)\n"
    "  any_system | umass_system            one relational query step: the\n"
    "  | nonumass_system | flagged_dx       sequenced analysis table\n"
    "  sisa_perct_cnt[_umass|_nonumass]     one aggregate node\n"
    "  conflict_list                        the cross-party MRN check\n"
    "  2a|2b|5a|5b|6a|6b[:population]       one fitted model; population is\n"
    "                                       any (default) | umass | nonumass";

bool ParseNode(const std::string& raw, NodeRequest& out, std::string& error) {
    if (raw.empty() || raw == "all") {
        out = NodeRequest{};
        return true;
    }

    out.name = raw;

    // step[:population]
    const size_t colon = raw.find(':');
    const std::string head = raw.substr(0, colon);
    const std::string tail = colon == std::string::npos ? std::string() : raw.substr(colon + 1);

    if (head == "conflict_list") {
        out.kind = NodeKind::ConflictList;
        return true;
    }

    // flagged_dx pass 2 IS the pooled sequencing, so it names the same table
    // any_system projects; the lineage page draws them as two nodes because the
    // projection drops columns, and nothing here drops any.
    if (head == "any_system" || head == "flagged_dx" || head == "umass_system" ||
        head == "nonumass_system") {
        out.kind = NodeKind::Relational;
        const std::string scope_name = head == "flagged_dx" ? "any_system" : head;
        if (!ParseScope(scope_name, out.scope)) {
            error = "unknown relational node '" + head + "'";
            return false;
        }
        return true;
    }

    if (head.rfind("sisa_perct_cnt", 0) == 0) {
        out.kind = NodeKind::Counts;
        const std::string suffix = head.substr(std::string("sisa_perct_cnt").size());
        if (suffix.empty()) {
            out.scope = SystemScope::Any;
        } else if (suffix == "_umass") {
            out.scope = SystemScope::UMass;
        } else if (suffix == "_nonumass") {
            out.scope = SystemScope::NonUMass;
        } else {
            error = "unknown aggregate node '" + head + "'";
            return false;
        }
        return true;
    }

    // Anything left has to be a model step.
    SystemScope scope = SystemScope::Any;
    if (!tail.empty() && !ParseScope(tail, scope)) {
        error = "unknown population '" + tail + "'; expected any, umass or nonumass";
        return false;
    }
    if (!ModelSpecFor(head, scope, out.spec, error)) return false;
    out.kind = NodeKind::Model;
    out.scope = scope;
    return true;
}

// =============================================================================
// Opening a relational query step to its owner
//
// harness.h:CohortFromSecure opens the same columns to EVERY party, which is
// what a cross-check between two oracles needs. This is the deployment form:
// the table goes to the one party entitled to it, and it reconstructs the two
// CLASS variables from their indicator columns so the CSV carries the table's
// real schema rather than a one-hot expansion of it.
// =============================================================================

PlainCohort OpenCohortToParty(const SecureCohort& c, int reveal_to, int party_id) {
    // Collective: every party reaches every open, whatever it gets back.
    const std::vector<DataType> subject = OpenRawToParty(c.subject_key, reveal_to, party_id);
    const std::vector<DataType> valid = OpenRawToParty(c.valid, reveal_to, party_id);
    const std::vector<DataType> dt = OpenRawToParty(c.encounter_dt, reveal_to, party_id);
    const std::vector<DataType> newage = OpenRawToParty(c.newage, reveal_to, party_id);
    const std::vector<DataType> index_visit = OpenRawToParty(c.index_visit, reveal_to, party_id);
    const std::vector<DataType> visit_num = OpenRawToParty(c.visit_num, reveal_to, party_id);
    const std::vector<DataType> final_visit = OpenRawToParty(c.final_visit, reveal_to, party_id);
    const std::vector<DataType> fu_month = OpenRawToParty(c.fu_month, reveal_to, party_id);
    const std::vector<DataType> fu_present = OpenRawToParty(c.fu_present, reveal_to, party_id);
    const std::vector<DataType> umass = OpenRawToParty(c.umass, reveal_to, party_id);
    const std::vector<DataType> sisa = OpenRawToParty(c.sisa, reveal_to, party_id);
    const std::vector<DataType> sa = OpenRawToParty(c.sa, reveal_to, party_id);

    std::vector<std::vector<DataType>> gender_is, hispanic_is;
    for (const AV& g : c.gender_is) gender_is.push_back(OpenRawToParty(g, reveal_to, party_id));
    for (const AV& h : c.hispanic_is)
        hispanic_is.push_back(OpenRawToParty(h, reveal_to, party_id));

    PlainCohort out;
    out.scope = c.scope;
    if (valid.empty()) return out;  // not this party's output

    const auto S = static_cast<DataType>(scale);
    for (size_t i = 0; i < valid.size(); ++i) {
        if (valid[i] != 1) continue;  // pad row
        out.subject_id.push_back(subject[i]);
        out.encounter_dt.push_back(dt[i]);
        out.newage.push_back(newage[i] / S);
        out.index_visit.push_back(index_visit[i]);
        out.visit_num.push_back(visit_num[i] / S);
        out.final_visit.push_back(final_visit[i]);
        out.fu_month.push_back(fu_month[i] / S);
        out.fu_month_present.push_back(fu_present[i]);
        out.data_source.push_back(umass[i] == 1 ? 1 : 2);
        out.sisa.push_back(sisa[i]);
        out.sa.push_back(sa[i]);

        // Back out the CLASS levels from the indicators. A row with no
        // indicator set cannot occur -- the levels partition the domain -- so
        // the fallthrough is the reference level, which is what a NULL would be
        // coded as upstream anyway.
        DataType gender = kGenderLevels.back();
        for (size_t k = 0; k < gender_is.size(); ++k)
            if (gender_is[k][i] == 1) gender = kGenderLevels[k];
        out.gender.push_back(gender);

        DataType hispanic = kHispanicLevels.back();
        for (size_t k = 0; k < hispanic_is.size(); ++k)
            if (hispanic_is[k][i] == 1) hispanic = kHispanicLevels[k];
        out.hispanic.push_back(hispanic);
    }
    return out;
}

// =============================================================================
// The result files
// =============================================================================

// Open a file for writing, complaining on stderr rather than failing silently.
// Returns false when the caller should skip writing.
bool OpenOutputFile(const std::string& path, std::ofstream& out) {
    out.open(path);
    if (!out) {
        std::cerr << "FATAL: could not open " << path << " for writing" << std::endl;
        return false;
    }
    return true;
}

// One relational query step: the analysis table, one row per encounter.
//
// Column names are the lineage's, not the struct's: `fu_month` is NULL rather
// than 0 where fu_month_present is 0, because that is what the source column
// holds and what every downstream b-axis model tests.
bool WriteCohortCsv(const std::string& path, const PlainCohort& c) {
    std::ofstream out;
    if (!OpenOutputFile(path, out)) return false;
    out << "table,subject_id,encounter_dt,newage,gender,hispanic,data_source,index_visit,"
           "visit_num,final_visit,fu_month,suicide_acutecare_icd_narrow,sa_icd_narrow\n";
    const char* table = ScopeName(c.scope);
    for (size_t i = 0; i < c.rows(); ++i) {
        out << table << ',' << c.subject_id[i] << ',' << c.encounter_dt[i] << ','
            << c.newage[i] << ',' << c.gender[i] << ',' << c.hispanic[i] << ','
            << c.data_source[i] << ',' << c.index_visit[i] << ',' << c.visit_num[i] << ','
            << c.final_visit[i] << ',';
        if (c.fu_month_present[i] == 1)
            out << c.fu_month[i];  // an empty field is the CSV spelling of NULL
        out << ',' << c.sisa[i] << ',' << c.sa[i] << '\n';
    }
    return true;
}

// One aggregate node: sisa_perct_cnt, one row per follow-up window.
bool WriteCountsCsv(const std::string& path, SystemScope scope,
                    const std::vector<SisaCounts>& rows) {
    std::ofstream out;
    if (!OpenOutputFile(path, out)) return false;
    out << "table,fu_month,pts_with_sisa,total_pts,sisa_pct,total_sa,null_fu_rows\n";
    const std::string suffix = scope == SystemScope::Any
                                   ? ""
                                   : (scope == SystemScope::UMass ? "_umass" : "_nonumass");
    for (const SisaCounts& r : rows)
        out << "sisa_perct_cnt" << suffix << ',' << r.window << ',' << r.pts_with_sisa << ','
            << r.total_pts << ',' << std::fixed << std::setprecision(4) << r.sisa_pct
            << std::defaultfloat << ',' << r.total_sa << ',' << r.null_fu_rows << '\n';
    return true;
}

// One fitted model: the coefficient table, plus the derived statistics that
// PrintFit shows and a final row for the variance component when there is one.
//
// z, p and the odds-ratio interval are deterministic public functions of
// (estimate, standard error), which the fit already discloses -- computing them
// here rather than under MPC protects nothing and costs nothing. `se` is empty
// where the fit declined to report one, which is not the same as zero.
bool WriteFitCsv(const std::string& path, const FitResult& fit, const ModelSpec& spec) {
    std::ofstream out;
    if (!OpenOutputFile(path, out)) return false;
    out << "step,population,term,estimate,std_error,z,p_value,odds_ratio,or_ci_lower,"
           "or_ci_upper,rows_used,converged\n";

    const char* population = fit.scope == SystemScope::Any
                                 ? "any_system"
                                 : (fit.scope == SystemScope::UMass ? "umass_system"
                                                                    : "nonumass_system");
    out << std::setprecision(9);
    for (size_t k = 0; k < fit.terms.size(); ++k) {
        const double est = fit.estimate[k];
        const bool have_se = std::isfinite(fit.se[k]) && fit.se[k] > 0.0;
        out << fit.step << ',' << population << ',' << fit.terms[k] << ',' << est << ',';
        if (have_se) {
            const double z = est / fit.se[k];
            out << fit.se[k] << ',' << z << ',' << TwoSidedNormalP(z) << ','
                << std::exp(est) << ',' << std::exp(est - kZ975 * fit.se[k]) << ','
                << std::exp(est + kZ975 * fit.se[k]);
        } else {
            out << ",,," << std::exp(est) << ",,";  // no SE: no z, no p, no interval
        }
        out << ',' << fit.rows_used << ',' << (fit.converged ? 1 : 0) << '\n';
    }
    if (fit.has_random)
        out << fit.step << ',' << population << ",random_intercept_variance," << fit.sigma2
            << ",,,,,,," << fit.rows_used << ',' << (fit.converged ? 1 : 0) << '\n';
    (void)spec;
    return true;
}

// conflict_list: one number, in the same CSV shape as everything else so a
// caller does not need a second parser for it.
bool WriteConflictCsv(const std::string& path, long conflicts) {
    std::ofstream out;
    if (!OpenOutputFile(path, out)) return false;
    out << "node,metric,value\n";
    out << "conflict_list,mrns_mapping_to_more_than_one_subject_id," << conflicts << '\n';
    return true;
}

}  // namespace cdough::regression

#pragma once

#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "./primitives.h"
#include "./segmented.h"

// The analysis cohort: the plaintext and secret-shared row layouts, the
// synthetic generator, and the per-party CSV ingestion path.

namespace cdough::regression {

// =============================================================================
// The analysis cohort
//
// One row per patient per acute-care encounter, flat, SORTED BY subject_id and
// padded to a power of two -- the layout the segmented scans above require. The
// twelve mixed models cluster on subject_id, and patients have a ragged number
// of encounters, so nothing here is padded per patient.
// =============================================================================

// Integer codes for the two CLASS variables. gender is varchar(20) in the source
// schema; MPC has no strings, so the ETL must hand over an integer code, and the
// mapping has to be pinned here because it decides which level is the reference.
//
// PROC GLIMMIX CLASS makes the LAST sorted level the reference. patsy and lme4
// use the FIRST. Getting this backwards silently flips the sign of every
// categorical coefficient, so the reference is stated explicitly rather than
// inherited from whatever the library happens to do.
const std::vector<DataType> kGenderLevels = {1, 2, 3};  // 3 (unknown/other) is the reference
const std::vector<DataType> kHispanicLevels = {0, 1};   // 1 is the reference, per SAS
const char* const kGenderNames[] = {"female", "male", "other/unknown"};

// fu_month only ever takes these values; anything else is NULL upstream.
const std::vector<DataType> kFuWindows = {1, 3, 6};

// Key sentinel for pad rows. Must not collide with a real subject_id, and must
// sort last so the pad block forms its own segment under both scan directions.
const DataType kKeySentinel = std::numeric_limits<DataType>::max();

// Which of the three source tables a cohort represents.
enum class SystemScope { Any, UMass, NonUMass };

const char* ScopeName(SystemScope s) {
    switch (s) {
        case SystemScope::Any: return "any_system";
        case SystemScope::UMass: return "umass_system";
        default: return "nonumass_system";
    }
}

const char* ScopeLabel(SystemScope s) {
    switch (s) {
        case SystemScope::Any: return "all systems";
        case SystemScope::UMass: return "UMass only";
        default: return "non-UMass only";
    }
}

// Plaintext form of one input table. Used to build the synthetic cohort, to
// parse CSV, and as the oracle the secure results are checked against.
struct PlainCohort {
    SystemScope scope = SystemScope::Any;
    std::vector<DataType> subject_id;
    std::vector<DataType> newage;
    std::vector<DataType> gender;
    std::vector<DataType> hispanic;
    std::vector<DataType> index_visit;
    std::vector<DataType> visit_num;
    std::vector<DataType> final_visit;
    std::vector<DataType> fu_month;          // 0/1/3/6; meaningless where present = 0
    std::vector<DataType> fu_month_present;  // 0 encodes SQL NULL
    std::vector<DataType> data_source;       // 1 = UMass, 2 = non-UMass
    std::vector<DataType> sisa;              // suicide_acutecare_icd_narrow, the outcome
    std::vector<DataType> sa;                // sa_icd_narrow

    size_t rows() const { return subject_id.size(); }

    // Sort by subject_id, then by visit_num within a patient. The segmented
    // scans compare adjacent keys, so grouping is only correct once sorted.
    void SortBySubject() {
        std::vector<size_t> order(rows());
        std::iota(order.begin(), order.end(), size_t{0});
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (subject_id[a] != subject_id[b]) return subject_id[a] < subject_id[b];
            return visit_num[a] < visit_num[b];
        });
        auto permute = [&](std::vector<DataType>& v) {
            std::vector<DataType> out(v.size());
            for (size_t i = 0; i < order.size(); ++i) out[i] = v[order[i]];
            v.swap(out);
        };
        permute(subject_id); permute(newage); permute(gender); permute(hispanic);
        permute(index_visit); permute(visit_num); permute(final_visit);
        permute(fu_month); permute(fu_month_present); permute(data_source);
        permute(sisa); permute(sa);
    }
};

// Secret-shared form. Flags and small integers are held as arithmetic 0/1 (or
// small ints) at precision 0; everything that feeds arithmetic is scaled.
struct SecureCohort {
    SystemScope scope = SystemScope::Any;
    size_t n = 0;      // real rows
    size_t n_pad = 0;  // padded to a power of two for the segmented scans

    BV subject_key;  // B-shared sort/group key, sentinel on pad rows
    std::vector<BV> keys;  // {subject_key}, the form the scan helpers take

    AV valid;         // 1 on real rows, 0 on pad rows (arithmetic)
    AV newage;        // scaled
    AV visit_num;     // scaled
    AV fu_month;      // scaled
    AV fu_present;    // 0/1
    AV index_visit;   // 0/1
    AV final_visit;   // 0/1
    AV sisa;          // 0/1
    AV sa;            // 0/1
    AV umass;         // 0/1, 1 where data_source == 1
    AV hispanic;      // 0/1 raw value (also used for the frequency table)
    std::vector<AV> gender_is;    // one 0/1 indicator per level in kGenderLevels
    std::vector<AV> hispanic_is;  // one 0/1 indicator per level in kHispanicLevels

    AV last_of_subject;   // 1 on the last row of each patient (arithmetic)
    AV first_of_subject;  // 1 on the first row of each patient (arithmetic)

    // Shared vectors carry an engine reference and so have no default
    // constructor; everything is sized to n_pad up front and filled in later.
    SecureCohort(EngineRef engine, size_t rows, size_t padded)
        : n(rows), n_pad(padded),
          subject_key(padded, engine),
          valid(padded, engine),
          newage(padded, engine),
          visit_num(padded, engine),
          fu_month(padded, engine),
          fu_present(padded, engine),
          index_visit(padded, engine),
          final_visit(padded, engine),
          sisa(padded, engine),
          sa(padded, engine),
          umass(padded, engine),
          hispanic(padded, engine),
          last_of_subject(padded, engine),
          first_of_subject(padded, engine) {}

    // Per-level group bits for the segmented scans. Built once per cohort and
    // reused by every objective evaluation of every model fitted to it.
    ScanPlan scan_plan;

    // Number of patients. Published: it is already the first reported column of
    // both descriptive nodes, so treating it as secret would protect nothing
    // while turning every quantile into an oblivious linear scan.
    size_t num_subjects = 0;
};

// =============================================================================
// Synthetic cohort
//
// Generated from a FIXED seed so every party derives an identical plaintext
// table. That matters for more than reproducibility: the number of encounters
// per patient is itself random, so a per-party RNG would give the parties
// different row counts and the shares would not line up. Party 0's copy is the
// one that gets secret shared; the others are used only as the oracle.
// =============================================================================

struct SyntheticTruth {
    double beta0 = 0.0;
    double beta_visit = 0.0;
    double beta_fu = 0.0;
    double beta_age = 0.0;
    double sigma = 0.0;
    double slope_diff = 0.0;  // the step 6 estimand: UMass vs non-UMass time slope
};

PlainCohort MakeSyntheticCohort(size_t num_subjects, SyntheticTruth& truth, uint64_t seed = 20260910ull) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> gauss(0.0, 1.0);

    truth.beta0 = -1.60;
    truth.beta_visit = 0.11;
    truth.beta_fu = 0.045;
    truth.beta_age = -0.012;
    truth.sigma = 0.70;
    truth.slope_diff = 0.06;

    PlainCohort c;
    c.scope = SystemScope::Any;

    for (size_t sid = 1; sid <= num_subjects; ++sid) {
        // Right-skewed visit count. A large share of patients have exactly one
        // encounter -- the lineage page notes those contribute no within-patient
        // information to any of the trend models, so the distribution matters.
        const double u = uni(rng);
        int visits = 1;
        if (u > 0.42) visits = 2;
        if (u > 0.66) visits = 3;
        if (u > 0.80) visits = 4;
        if (u > 0.88) visits = 5 + static_cast<int>(uni(rng) * 4);
        if (u > 0.97) visits = 9 + static_cast<int>(uni(rng) * 8);

        const DataType gender = kGenderLevels[static_cast<size_t>(uni(rng) * 3.0) % 3];
        const DataType hispanic = uni(rng) < 0.12 ? 1 : 0;
        const int base_age = 14 + static_cast<int>(uni(rng) * 55.0);
        const double u_subject = truth.sigma * gauss(rng);
        // A patient's encounters mostly stay inside one system.
        const double umass_propensity = uni(rng) < 0.55 ? 0.85 : 0.15;

        for (int v = 1; v <= visits; ++v) {
            const bool is_umass = uni(rng) < umass_propensity;
            const DataType data_source = is_umass ? 1 : 2;

            // fu_month is null outside the windows and on a same-day repeat.
            DataType fu = 0, fu_present = 1;
            if (v == 1) {
                fu = 0;
            } else {
                const double f = uni(rng);
                if (f < 0.28) fu = 1;
                else if (f < 0.58) fu = 3;
                else if (f < 0.82) fu = 6;
                else fu_present = 0;  // beyond 182 days, or a same-day repeat
            }

            const double time_visit = static_cast<double>(v);
            const double time_fu = static_cast<double>(fu);
            const double age = base_age + (v - 1) * 0.4;
            const double slope = truth.beta_visit + (is_umass ? truth.slope_diff : 0.0);
            const double eta = truth.beta0 + slope * time_visit +
                               truth.beta_fu * time_fu + truth.beta_age * (age - 40.0) +
                               (gender == 1 ? 0.18 : 0.0) + (hispanic == 1 ? 0.10 : 0.0) +
                               u_subject;
            const double prob = 1.0 / (1.0 + std::exp(-eta));
            const DataType sisa = uni(rng) < prob ? 1 : 0;
            const DataType sa = (sisa == 1 && uni(rng) < 0.35) ? 1 : 0;

            c.subject_id.push_back(static_cast<DataType>(sid));
            c.newage.push_back(static_cast<DataType>(std::llround(age)));
            c.gender.push_back(gender);
            c.hispanic.push_back(hispanic);
            c.index_visit.push_back(v == 1 ? 1 : 0);
            c.visit_num.push_back(v);
            c.final_visit.push_back(v == visits ? 1 : 0);
            c.fu_month.push_back(fu);
            c.fu_month_present.push_back(fu_present);
            c.data_source.push_back(data_source);
            c.sisa.push_back(sisa);
            c.sa.push_back(sa);
        }
    }
    c.SortBySubject();
    return c;
}

// Derive umass_system / nonumass_system from any_system, the way the upstream
// ETL does: filter on data_source, then RE-SEQUENCE the per-system columns,
// because a patient's third encounter overall may be their first in this system.
//
// The `reproduce_etl_bug` flag deliberately reproduces the fault documented on
// the lineage page: the ETL sets fu_month<sfx> = 0 by testing the GLOBAL
// index_visit rather than index_visit<sfx>, so a patient whose history in this
// system starts later than their overall history gets NULL on their first row
// here. It is on by default because the point of the port is to match the
// pipeline as it actually runs, not as it was meant to.
PlainCohort DeriveSystemCohort(const PlainCohort& any, SystemScope scope,
                               bool reproduce_etl_bug = true) {
    const DataType want = (scope == SystemScope::UMass) ? 1 : 2;
    PlainCohort c;
    c.scope = scope;

    // Rows for this system, in the order they already appear (sorted by subject).
    std::vector<size_t> rows;
    for (size_t i = 0; i < any.rows(); ++i)
        if (any.data_source[i] == want) rows.push_back(i);

    size_t i = 0;
    while (i < rows.size()) {
        size_t j = i;
        while (j < rows.size() && any.subject_id[rows[j]] == any.subject_id[rows[i]]) ++j;
        const size_t count = j - i;
        for (size_t k = 0; k < count; ++k) {
            const size_t src = rows[i + k];
            const int v = static_cast<int>(k) + 1;

            DataType fu = any.fu_month[src];
            DataType fu_present = any.fu_month_present[src];
            if (v == 1) {
                if (reproduce_etl_bug && any.index_visit[src] != 1) {
                    // First encounter in THIS system, but not the patient's first
                    // overall: the upstream guard misses it and leaves it null.
                    fu = 0;
                    fu_present = 0;
                } else {
                    fu = 0;
                    fu_present = 1;
                }
            }

            c.subject_id.push_back(any.subject_id[src]);
            c.newage.push_back(any.newage[src]);
            c.gender.push_back(any.gender[src]);
            c.hispanic.push_back(any.hispanic[src]);
            c.index_visit.push_back(v == 1 ? 1 : 0);
            c.visit_num.push_back(v);
            c.final_visit.push_back(k + 1 == count ? 1 : 0);
            c.fu_month.push_back(fu);
            c.fu_month_present.push_back(fu_present);
            c.data_source.push_back(any.data_source[src]);
            c.sisa.push_back(any.sisa[src]);
            c.sa.push_back(any.sa[src]);
        }
        i = j;
    }
    return c;
}

// =============================================================================
// Ingestion
//
// Each of the three tables has a designated input party that holds it in the
// clear, mirroring examples/ex6_three_party_private_input.cpp and
// docker/DEPLOYMENT.md: every party calls the loader, only the owner opens the
// file, and plaintext never crosses the wire.
//
// Because the owner holds the table, the sort by subject_id and the categorical
// indicator coding are done locally in plaintext before sharing. That reveals
// nothing the owner does not already know and avoids an oblivious sort over the
// whole table. A deployment in which NO party holds any_system in the clear
// would need an oblivious sort and merge instead; that is out of scope here.
// =============================================================================

// Column order of the CSV. Bracketed names mark the columns that would be
// B-shared under the EncodedTable convention; kept here for documentation and
// so the header check is exact.
const std::vector<std::string> kCsvHeader = {
    "[subject_id]", "newage",      "[gender]",       "[hispanic]",
    "[index_visit]", "visit_num",  "[final_visit]",  "[fu_month]",
    "[fu_month_present]", "[data_source]", "[sisa]", "[sa]"};

std::string CohortCsvName(SystemScope scope) {
    return std::string(ScopeName(scope)) + ".csv";
}

// Write a cohort out in the exact format the loader expects. Used to dump the
// synthetic cohort for the R / Python / SQLite cross-checks.
void WriteCohortCsv(const PlainCohort& c, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "could not open " << path << " for writing" << std::endl;
        return;
    }
    for (size_t i = 0; i < kCsvHeader.size(); ++i)
        out << kCsvHeader[i] << (i + 1 == kCsvHeader.size() ? '\n' : ',');
    for (size_t i = 0; i < c.rows(); ++i) {
        out << c.subject_id[i] << ',' << c.newage[i] << ',' << c.gender[i] << ','
            << c.hispanic[i] << ',' << c.index_visit[i] << ',' << c.visit_num[i] << ','
            << c.final_visit[i] << ',' << c.fu_month[i] << ',' << c.fu_month_present[i] << ','
            << c.data_source[i] << ',' << c.sisa[i] << ',' << c.sa[i] << '\n';
    }
}

// Read a cohort CSV. Header order is free -- columns are matched by name, as
// EncodedTable::inputCSVTableData does -- but every expected name must be
// present. Values are integers only.
PlainCohort ReadCohortCsv(const std::string& path, SystemScope scope) {
    PlainCohort c;
    c.scope = scope;
    std::ifstream in(path);
    if (!in) {
        std::cerr << "FATAL: could not open " << path << std::endl;
        std::exit(1);
    }

    std::string line;
    if (!std::getline(in, line)) {
        std::cerr << "FATAL: " << path << " is empty" << std::endl;
        std::exit(1);
    }

    std::vector<std::string> header;
    {
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            while (!tok.empty() && (tok.back() == '\r' || tok.back() == ' ')) tok.pop_back();
            header.push_back(tok);
        }
    }
    std::map<std::string, size_t> pos;
    for (size_t i = 0; i < header.size(); ++i) pos[header[i]] = i;
    for (const std::string& want : kCsvHeader) {
        if (pos.find(want) == pos.end()) {
            std::cerr << "FATAL: column " << want << " missing from " << path << std::endl;
            std::exit(1);
        }
    }

    auto* const targets = new std::vector<DataType>*[kCsvHeader.size()]{
        &c.subject_id, &c.newage, &c.gender, &c.hispanic,
        &c.index_visit, &c.visit_num, &c.final_visit, &c.fu_month,
        &c.fu_month_present, &c.data_source, &c.sisa, &c.sa};

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) fields.push_back(tok);
        for (size_t k = 0; k < kCsvHeader.size(); ++k) {
            const size_t idx = pos[kCsvHeader[k]];
            targets[k]->push_back(idx < fields.size()
                                      ? static_cast<DataType>(std::stoll(fields[idx]))
                                      : DataType{0});
        }
    }
    delete[] targets;

    c.SortBySubject();
    return c;
}

// Secret share a plaintext cohort from `input_party`. Non-owning parties pass a
// PlainCohort of the same shape with zero contents; only the row count and the
// schema have to agree across parties.
SecureCohort ShareCohort(EngineRef engine, const PlainCohort& c, int input_party) {
    const size_t n = c.rows();
    const size_t np = NextPowerOfTwo(std::max<size_t>(n, 2));
    SecureCohort sc(engine, n, np);
    sc.scope = c.scope;

    {
        std::set<DataType> subjects(c.subject_id.begin(), c.subject_id.end());
        sc.num_subjects = subjects.size();
    }

    // Pad rows carry the key sentinel (so they form their own segment under both
    // scan directions) and zeros everywhere else (so they contribute nothing).
    auto share_int = [&](const std::vector<DataType>& src) {
        cdough::Vector<DataType> v(np, 0);
        for (size_t i = 0; i < np; ++i) v[i] = i < n ? src[i] : DataType{0};
        AV out = engine.template secret_share_a<DataType>(v, input_party, 0);
        out.setPrecision(0);
        return out;
    };
    auto share_scaled = [&](const std::vector<DataType>& src) {
        cdough::Vector<DataType> v(np, precision);
        for (size_t i = 0; i < np; ++i)
            v[i] = i < n ? static_cast<DataType>(src[i]) * DataType(scale) : DataType{0};
        AV out = engine.template secret_share_a<DataType>(v, input_party, precision);
        out.setPrecision(0);
        return out;
    };
    auto share_indicator = [&](const std::vector<DataType>& src, DataType level) {
        cdough::Vector<DataType> v(np, 0);
        for (size_t i = 0; i < np; ++i) v[i] = (i < n && src[i] == level) ? 1 : 0;
        AV out = engine.template secret_share_a<DataType>(v, input_party, 0);
        out.setPrecision(0);
        return out;
    };

    {
        cdough::Vector<DataType> kv(np, 0);
        for (size_t i = 0; i < np; ++i) kv[i] = i < n ? c.subject_id[i] : kKeySentinel;
        sc.subject_key = engine.template secret_share_b<DataType>(kv, input_party);
    }
    sc.keys.push_back(sc.subject_key);

    {
        cdough::Vector<DataType> vv(np, 0);
        for (size_t i = 0; i < np; ++i) vv[i] = i < n ? 1 : 0;
        sc.valid = engine.template secret_share_a<DataType>(vv, input_party, 0);
        sc.valid.setPrecision(0);
    }

    sc.newage = share_scaled(c.newage);
    sc.visit_num = share_scaled(c.visit_num);
    sc.fu_month = share_scaled(c.fu_month);
    sc.fu_present = share_int(c.fu_month_present);
    sc.index_visit = share_int(c.index_visit);
    sc.final_visit = share_int(c.final_visit);
    sc.sisa = share_int(c.sisa);
    sc.sa = share_int(c.sa);
    sc.hispanic = share_int(c.hispanic);
    sc.umass = share_indicator(c.data_source, 1);

    for (DataType lvl : kGenderLevels) sc.gender_is.push_back(share_indicator(c.gender, lvl));
    for (DataType lvl : kHispanicLevels)
        sc.hispanic_is.push_back(share_indicator(c.hispanic, lvl));

    // Group-boundary indicators, computed once and reused by every node. The
    // AND with `valid` is what keeps the sentinel pad block from contributing a
    // spurious final group.
    AV last = LastOfGroupArith(sc.keys);
    sc.last_of_subject = *(last * sc.valid);
    sc.last_of_subject.setPrecision(0);

    BV first_b = FirstOfGroup(sc.keys);
    AV first = *(first_b.b2a_bit());
    first.setPrecision(0);
    sc.first_of_subject = *(first * sc.valid);
    sc.first_of_subject.setPrecision(0);

    sc.scan_plan = BuildScanPlan(sc.subject_key);

    return sc;
}


}  // namespace cdough::regression

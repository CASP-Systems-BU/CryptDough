#pragma once

#include <algorithm>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "./library/primitives.h"
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
    // Encounter date as integer days from a fixed public epoch. This is the
    // ordering key the sequencing is defined on; upstream it is
    // admit_date_pcc / encounter_dt.
    std::vector<DataType> encounter_dt;
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
    AV encounter_dt;  // days from the public epoch, the ordering key
    AV visit_num;     // scaled
    AV fu_month;      // scaled
    AV fu_present;    // 0/1
    AV index_visit;   // 0/1
    AV final_visit;   // 0/1
    AV sisa;          // 0/1
    AV sa;            // 0/1
    AV umass;         // 0/1, 1 where data_source == 1
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
          encounter_dt(padded, engine),
          visit_num(padded, engine),
          fu_month(padded, engine),
          fu_present(padded, engine),
          index_visit(padded, engine),
          final_visit(padded, engine),
          sisa(padded, engine),
          sa(padded, engine),
          umass(padded, engine),
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
// Moving a cohort onto another engine (semantic task 0024)
//
// The model fits can run concurrently on several engines (-neng), but a shared
// vector is bound to the engine that made it, and every operation on it runs on
// that engine's workers and communicators. So each extra engine gets its own
// deep copy of the cohorts, built here once before any fit thread starts.
//
// This is a local copy with no communication. SharedVector::operator= copies the
// contents element-wise and keeps the DESTINATION's engine, which is the same
// assignment Clone() relies on. A replicated-3PC share depends only on the
// party ID, and that is the same for every engine in this process, so the
// copies are valid shares under the new engine. Protocols that bind per-engine
// state into a share (MACs, OT correlations) would not have this property --
// the driver refuses -neng > 1 on them.
// =============================================================================

AV CloneOnto(const AV& v, EngineRef engine) {
    AV out(v.size(), engine);
    out = v;
    out.setPrecision(v.getPrecision());
    return out;
}

BV CloneOnto(const BV& v, EngineRef engine) {
    BV out(v.size(), engine);
    out = v;
    out.setPrecision(v.getPrecision());
    return out;
}

template <typename Vec>
std::vector<Vec> CloneOnto(const std::vector<Vec>& vs, EngineRef engine) {
    std::vector<Vec> out;
    out.reserve(vs.size());
    for (const Vec& v : vs) out.push_back(CloneOnto(v, engine));
    return out;
}

SecureCohort CloneCohortOnto(const SecureCohort& c, EngineRef engine) {
    SecureCohort out(engine, c.n, c.n_pad);
    out.scope = c.scope;
    out.subject_key = CloneOnto(c.subject_key, engine);
    out.keys = CloneOnto(c.keys, engine);
    out.valid = CloneOnto(c.valid, engine);
    out.newage = CloneOnto(c.newage, engine);
    out.encounter_dt = CloneOnto(c.encounter_dt, engine);
    out.visit_num = CloneOnto(c.visit_num, engine);
    out.fu_month = CloneOnto(c.fu_month, engine);
    out.fu_present = CloneOnto(c.fu_present, engine);
    out.index_visit = CloneOnto(c.index_visit, engine);
    out.final_visit = CloneOnto(c.final_visit, engine);
    out.sisa = CloneOnto(c.sisa, engine);
    out.sa = CloneOnto(c.sa, engine);
    out.umass = CloneOnto(c.umass, engine);
    out.gender_is = CloneOnto(c.gender_is, engine);
    out.hispanic_is = CloneOnto(c.hispanic_is, engine);
    out.last_of_subject = CloneOnto(c.last_of_subject, engine);
    out.first_of_subject = CloneOnto(c.first_of_subject, engine);
    out.scan_plan.n = c.scan_plan.n;
    out.scan_plan.depth = c.scan_plan.depth;
    out.scan_plan.forward = CloneOnto(c.scan_plan.forward, engine);
    out.scan_plan.reverse = CloneOnto(c.scan_plan.reverse, engine);
    out.num_subjects = c.num_subjects;
    return out;
}

// Everything downstream of PlainCohort -- the synthetic generator, the
// three-table CSV schema, and ShareCohort -- was removed with task 0011. The
// pipeline now starts at cdrcatsse_match_pcc (etl.h) and reaches shares
// through the two-owner merge (secure.h), so a single-owner per-table import
// no longer describes anything the pipeline does.
//
// PlainCohort itself stays: SequencePlain (secure.h) builds one as the oracle
// the fits are scored against in reporting.h.

}  // namespace cdough::regression

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "./primitives.h"

// The upstream ETL, ported from
//   https://cs-people.bu.edu/liagos/pilot/pcc_sisa_lineage.html
// which documents PCC_SISA_data_for_BU.sas.
//
// Lineage nodes, in order:
//   cdrcatsse_match_pcc  base table -- "the only real input to the pipeline"
//   conflict_list        MRNs mapping to more than one subject_id
//   pcc_mrn              the same rows with two MRN conflicts repaired
//   flagged_dx  pass 1   filter to Emergency / Acute Care, parse the suicide
//                        flags out of the concatenated diagnosis string, and
//                        set data_source from the facility name
//   flagged_dx  pass 2   visit sequencing within patient  -> secure/, not here
//   umass / nonumass     per-system re-sequencing         -> secure/, not here
//   any_system, umass_system, nonumass_system   projections
//
// WHAT LIVES IN THIS FILE, AND WHY
//
// We simulate a world in which the base table is split across two data owners
// whose patients overlap. Every node above divides cleanly into two classes:
//
//   ROW-LOCAL   pcc_mrn and flagged_dx pass 1. Each output column is a function
//               of ONE input row -- a string match, a date cast, a lookup. An
//               owner can compute these over its own half in the clear. That
//               reveals nothing the owner does not already hold, and it keeps
//               every LIKE '%...%' out of MPC, where string matching would
//               dominate the cost of the entire pipeline. This file is that
//               half, and it runs at each owner before anything is shared.
//
//   CROSS-PARTY conflict_list and both sequencing passes. These partition by
//               subject_id or pat_mrn, and a patient's encounters live at BOTH
//               owners, so no owner can compute them alone. They are in
//               secure.h and run under MPC on the merged table.

namespace cdough::regression {

// =============================================================================
// cdrcatsse_match_pcc -- the base table
//
// Already a join result rather than a source table: PointClickCare encounter
// records matched to CDR-CATSSE study records on medical record number, with
// the CDR demographics denormalised onto every encounter row.
//
// The lineage warns that there is no encounter key anywhere in either script,
// so nothing can assert the grain. We keep admit_date at day resolution and
// treat (subject_id, admit_date, facility) as the de-facto key.
// =============================================================================

struct PlainBaseTable {
    std::vector<std::string> pat_mrn;             // varchar(9), zero-padded
    std::vector<DataType> subject_id;             // de-identified study ID
    std::vector<std::string> visit_type_pcc;      // only two values survive
    std::vector<std::string> visit_major_type_pccc;  // QC only, never in logic
    std::vector<DataType> admit_date_pcc;         // days from a public epoch
    std::vector<std::string> diagnoses_pcc;       // concatenated ICD + free text
    std::vector<std::string> visit_facility_pcc;  // matched against nine sites
    std::vector<DataType> dob;                    // days from the same epoch
    std::vector<DataType> gender_pcc;
    std::vector<DataType> hispanic;

    size_t rows() const { return subject_id.size(); }

    void PushRow(const std::string& mrn, DataType sid, const std::string& vtype,
                 const std::string& vmajor, DataType admit, const std::string& dx,
                 const std::string& facility, DataType born, DataType gender, DataType hisp) {
        pat_mrn.push_back(mrn);
        subject_id.push_back(sid);
        visit_type_pcc.push_back(vtype);
        visit_major_type_pccc.push_back(vmajor);
        admit_date_pcc.push_back(admit);
        diagnoses_pcc.push_back(dx);
        visit_facility_pcc.push_back(facility);
        dob.push_back(born);
        gender_pcc.push_back(gender);
        hispanic.push_back(hisp);
    }
};

// The nine UMass Memorial sites. data_source = 2 is the ELSE branch of this
// comparison, which the lineage flags as a hazard: a renamed facility, a
// trailing space or a new UMass site silently becomes non-UMass.
const std::vector<std::string> kUMassFacilities = {
    "UMass Memorial - Leominster Urgent Care",
    "UMass Memorial Harrington Hospital - Southbridge Campus",
    "UMass Memorial Harrington Hospital - Webster Campus",
    "UMass Memorial Health - Marlborough Hospital",
    "UMass Memorial Healthalliance - Clinton Hospital - Clinton Campus",
    "UMass Memorial Healthalliance - Clinton Hospital - Leominster Campus",
    "UMass Memorial Medical Center",
    "UMass Memorial Medical Center - Memorial Campus",
    "UMass Memorial Medical Center - University Campus"};

const std::vector<std::string> kNonUMassFacilities = {
    "Saint Vincent Hospital", "Milford Regional Medical Center",
    "Heywood Hospital", "Emerson Hospital", "Baystate Medical Center"};

// Only these two visit types survive the filter in flagged_dx.
const std::vector<std::string> kKeptVisitTypes = {"Emergency", "Acute Care"};

// =============================================================================
// conflict_list -- plaintext reference implementation
//
// MRNs that map to more than one study ID. One physical patient carrying two
// subject_id values would be counted twice in every analysis and would have
// their visit history split in half.
//
// Upstream this is a dead end: nothing reads it, and the repair was typed into
// pcc_mrn by hand as two literal MRNs. In the two-owner world it stops being a
// dead end, because an MRN can carry one study ID at each owner and neither can
// see the disagreement alone. This function is the ORACLE the secure version in
// secure.h is checked against; see SecureConflictList.
// =============================================================================

struct MrnConflict {
    std::string pat_mrn;
    long unique_id_count = 0;
};

std::vector<MrnConflict> ConflictList(const PlainBaseTable& base) {
    std::map<std::string, std::set<DataType>> ids;
    for (size_t i = 0; i < base.rows(); ++i) ids[base.pat_mrn[i]].insert(base.subject_id[i]);

    std::vector<MrnConflict> out;
    for (const auto& [mrn, set] : ids)
        if (set.size() > 1) out.push_back(MrnConflict{mrn, static_cast<long>(set.size())});
    return out;
}

// =============================================================================
// pcc_mrn -- the MRN repair
//
// A full copy of the base table with the two MRN-to-study-ID conflicts
// resolved. No rows are added or dropped.
//
// The lineage notes two faults, both reproduced here because the point of the
// port is to match the pipeline as it runs:
//   - The repair DISCARDS rather than merges: overwriting subject_id throws
//     away whichever study ID was previously attached to those encounters.
//   - Detection and repair can drift. The fix is hard-coded rather than driven
//     from conflict_list, so a data refresh that introduces a third conflict is
//     reported by ConflictList and silently ignored here.
// =============================================================================

const std::map<std::string, DataType> kMrnRepairs = {{"000852319", 1040}, {"000215487", 1076}};

// Row-local: one output row per input row, and the new subject_id depends only
// on this row's pat_mrn. Each owner applies it to its own half.
void ApplyMrnRepair(PlainBaseTable& t) {
    for (size_t i = 0; i < t.rows(); ++i) {
        const auto it = kMrnRepairs.find(t.pat_mrn[i]);
        if (it != kMrnRepairs.end()) t.subject_id[i] = it->second;
    }
}

// =============================================================================
// flagged_dx pass 1 -- flags parsed from the diagnosis string
//
// "The centre of the pipeline, and where every analysis variable is actually
// born." Keeps only Emergency and Acute Care encounters, parses the suicide
// ideation and attempt flags out of the concatenated diagnosis string, and sets
// data_source from the facility name.
//
// Every column here is a function of a single row, so this runs at the owner.
// =============================================================================

struct PlainFlagged {
    std::vector<std::string> pat_mrn;
    std::vector<DataType> subject_id;
    std::vector<DataType> encounter_dt;  // datepart(admit_date_pcc)
    std::vector<DataType> newage;        // completed years at the encounter
    std::vector<DataType> gender;
    std::vector<DataType> hispanic;
    std::vector<DataType> data_source;  // 1 = UMass, 2 = non-UMass

    // The nine primary flags.
    std::vector<DataType> si_dx_pcc;
    std::vector<DataType> si_icd_pcc;
    std::vector<DataType> si_prob_dx_pcc;  // always 0 as written -- see below
    std::vector<DataType> sa_dx_pcc;
    std::vector<DataType> sa_icd_narrow_pcc;
    std::vector<DataType> sa_dx_unintent_pcc;
    std::vector<DataType> sa_icd_unintent_pcc;
    std::vector<DataType> sa_dx_undetermine_pcc;
    std::vector<DataType> sa_icd_undetermine_pcc;

    // The composites.
    std::vector<DataType> si_prob_dx_pcc_fixed;
    std::vector<DataType> si_dx_icd_pcc;
    std::vector<DataType> sisa_icd_narrow_pcc;
    std::vector<DataType> sisa_icd_broad_unintent_pcc;
    std::vector<DataType> acute_care_related_sisa_pcc;
    std::vector<DataType> sa_broad_unintent_pcc;
    std::vector<DataType> sa_broad_unintent_undeterm_pcc;
    std::vector<DataType> suicide_acutecare_icd_narrow;  // THE outcome variable
    std::vector<DataType> suicide_acutecare_icd_broad;

    size_t rows() const { return subject_id.size(); }
};

std::string UpperCase(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return out;
}

// SQL LIKE '%needle%' -- an unanchored substring test.
bool Contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

bool ContainsAny(const std::string& haystack, const std::vector<std::string>& needles) {
    for (const std::string& n : needles)
        if (Contains(haystack, n)) return true;
    return false;
}

// `reproduce_etl_bugs` keeps the two documented pass-1 faults. On by default,
// for the same reason DeriveSystemCohort keeps its own: the port should match
// the pipeline as it actually runs, not as it was meant to.
PlainFlagged DeriveFlagged(const PlainBaseTable& t, bool reproduce_etl_bugs = true) {
    PlainFlagged f;
    for (size_t i = 0; i < t.rows(); ++i) {
        // WHERE visit_type_pcc IN ('Emergency', 'Acute Care')
        if (std::find(kKeptVisitTypes.begin(), kKeptVisitTypes.end(), t.visit_type_pcc[i]) ==
            kKeptVisitTypes.end())
            continue;

        const std::string up = UpperCase(t.diagnoses_pcc[i]);

        // BUG (lineage): index(up_text, "SI") is an unanchored substring match,
        // so it fires on DEPRESSION, HYPERTENSION, CONTUSION, LESION and many
        // other ordinary diagnoses. Negated mentions are not screened either.
        const std::vector<std::string> si_text =
            reproduce_etl_bugs
                ? std::vector<std::string>{"IDEATION", "SI", "SUICIDAL IDEATIONS",
                                           "SUICIDAL THOUGHTS"}
                : std::vector<std::string>{"IDEATION", "SUICIDAL IDEATIONS",
                                           "SUICIDAL THOUGHTS"};
        const DataType si_dx = ContainsAny(up, si_text) ? 1 : 0;
        const DataType si_icd = Contains(up, "R45851") ? 1 : 0;
        const DataType sa_dx =
            ContainsAny(up, {"SUICIDE ATTEMPT", "INTENTIONAL SELF-HARM", "SUICIDAL WITH PLAN"})
                ? 1
                : 0;
        const DataType sa_icd_narrow = ContainsAny(up, {"X781XXA", "T50902A", "T518X2A",
                                                        "T1491XA", "T424X2A", "T426X2A"})
                                           ? 1
                                           : 0;
        const DataType sa_dx_unintent = Contains(up, "UNINTENTIONAL") ? 1 : 0;
        const DataType sa_icd_unintent =
            ContainsAny(up, {"T50901A", "T424X1A", "T383X1A", "T40411A", "T63301A"}) ? 1 : 0;
        const DataType sa_dx_undet = Contains(up, "UNDETERMINED") ? 1 : 0;
        const DataType sa_icd_undet = ContainsAny(up, {"T402X4A", "T50904A"}) ? 1 : 0;

        // BUG (lineage): si_prob_dx_pcc tests sa_dx_pcc = 0, but sa_dx_pcc is
        // not assigned until 7 lines later and is not retained, so it is
        // missing at that point and the comparison is false on every row. The
        // SQL port writes the literal 0 and carries the fixed version beside
        // it. This silently affects si_dx_icd_pcc and acute_care_related_sisa.
        const DataType si_prob_fixed = (Contains(up, "SUICIDAL") && sa_dx == 0) ? 1 : 0;
        const DataType si_prob = reproduce_etl_bugs ? 0 : si_prob_fixed;

        const bool is_umass = std::find(kUMassFacilities.begin(), kUMassFacilities.end(),
                                        t.visit_facility_pcc[i]) != kUMassFacilities.end();

        f.pat_mrn.push_back(t.pat_mrn[i]);
        f.subject_id.push_back(t.subject_id[i]);
        f.encounter_dt.push_back(t.admit_date_pcc[i]);
        // FLOOR(DATE_PART('year', AGE(encounter_dt, dob)))
        f.newage.push_back(static_cast<DataType>((t.admit_date_pcc[i] - t.dob[i]) / 365));
        f.gender.push_back(t.gender_pcc[i]);
        f.hispanic.push_back(t.hispanic[i]);
        f.data_source.push_back(is_umass ? 1 : 2);

        f.si_dx_pcc.push_back(si_dx);
        f.si_icd_pcc.push_back(si_icd);
        f.si_prob_dx_pcc.push_back(si_prob);
        f.sa_dx_pcc.push_back(sa_dx);
        f.sa_icd_narrow_pcc.push_back(sa_icd_narrow);
        f.sa_dx_unintent_pcc.push_back(sa_dx_unintent);
        f.sa_icd_unintent_pcc.push_back(sa_icd_unintent);
        f.sa_dx_undetermine_pcc.push_back(sa_dx_undet);
        f.sa_icd_undetermine_pcc.push_back(sa_icd_undet);

        f.si_prob_dx_pcc_fixed.push_back(si_prob_fixed);
        f.si_dx_icd_pcc.push_back((si_dx || si_icd) ? 1 : 0);
        f.sisa_icd_narrow_pcc.push_back((si_icd || sa_icd_narrow) ? 1 : 0);
        f.sisa_icd_broad_unintent_pcc.push_back(
            (si_icd || sa_icd_narrow || sa_icd_unintent) ? 1 : 0);
        f.acute_care_related_sisa_pcc.push_back(
            (si_dx || si_icd || sa_dx || sa_icd_narrow) ? 1 : 0);
        f.sa_broad_unintent_pcc.push_back(
            (sa_dx || sa_icd_narrow || sa_dx_unintent || sa_icd_unintent) ? 1 : 0);
        f.sa_broad_unintent_undeterm_pcc.push_back(
            (sa_dx || sa_icd_narrow || sa_dx_unintent || sa_icd_unintent || sa_dx_undet ||
             sa_icd_undet)
                ? 1
                : 0);
        // suicide_acutecare_icd_narrow is a copy of sisa_icd_narrow_pcc.
        f.suicide_acutecare_icd_narrow.push_back((si_icd || sa_icd_narrow) ? 1 : 0);
        f.suicide_acutecare_icd_broad.push_back(
            (si_icd || sa_icd_narrow || sa_icd_unintent || sa_icd_undet) ? 1 : 0);
    }
    return f;
}

// =============================================================================
// Synthetic cdrcatsse_match_pcc
//
// Generates the BASE table -- text and all -- rather than the analysis cohort,
// so the ETL above is exercised for real: the flags are parsed out of strings,
// data_source is decided by a facility name, and the documented faults have
// something to fire on.
// =============================================================================

struct GenTruth {
    double beta0 = -1.1;
    double beta_visit = 0.08;
    double beta_fu = 0.045;
    double beta_age = -0.012;
    double sigma = 0.70;
    double slope_diff = 0.06;
};

// Ordinary diagnoses that contain "SI" as a substring. These exist so the
// unanchored-SI fault has something to misfire on, which is the only way a port
// can demonstrate the bug the lineage documents.
const std::vector<std::string> kDecoyDiagnoses = {
    "MAJOR DEPRESSION, RECURRENT", "ESSENTIAL HYPERTENSION", "SCALP CONTUSION",
    "SKIN LESION OF FOREARM", "ACUTE SINUSITIS", "OBESITY, UNSPECIFIED"};

const std::vector<std::string> kPlainDiagnoses = {
    "ACUTE PHARYNGITIS", "CLOSED FRACTURE OF WRIST", "ABDOMINAL PAIN, UNSPECIFIED",
    "TYPE 2 DIABETES MELLITUS", "ASTHMA EXACERBATION"};

PlainBaseTable GenerateBaseTable(size_t num_subjects, GenTruth& truth, unsigned seed = 20260911) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> gauss(0.0, 1.0);

    PlainBaseTable t;

    for (size_t sid = 1; sid <= num_subjects; ++sid) {
        const double u = uni(rng);
        int visits = 1;
        if (u > 0.42) visits = 2;
        if (u > 0.66) visits = 3;
        if (u > 0.80) visits = 4;
        if (u > 0.88) visits = 5 + static_cast<int>(uni(rng) * 4);
        if (u > 0.97) visits = 9 + static_cast<int>(uni(rng) * 8);

        const DataType gender = static_cast<DataType>(uni(rng) * 3.0) % 3 + 1;
        const DataType hispanic = uni(rng) < 0.12 ? 1 : 0;
        const DataType born = -static_cast<DataType>((14 + uni(rng) * 55.0) * 365);
        const double u_subject = truth.sigma * gauss(rng);
        const double umass_propensity = uni(rng) < 0.55 ? 0.85 : 0.15;

        // A stable MRN per patient, zero-padded to nine characters.
        std::string mrn = std::to_string(100000 + sid * 7 % 900000);
        while (mrn.size() < 9) mrn.insert(mrn.begin(), '0');

        // Encounter dates first; the sequencing is derived from them downstream.
        const DataType index_dt = static_cast<DataType>(uni(rng) * 3650.0);
        std::vector<DataType> gaps{0};
        for (int v = 2; v <= visits; ++v) {
            const double f = uni(rng);
            if (f < 0.28) gaps.push_back(1 + static_cast<DataType>(uni(rng) * 30.0));
            else if (f < 0.58) gaps.push_back(31 + static_cast<DataType>(uni(rng) * 61.0));
            else if (f < 0.82) gaps.push_back(92 + static_cast<DataType>(uni(rng) * 91.0));
            else gaps.push_back(183 + static_cast<DataType>(uni(rng) * 400.0));
        }
        std::sort(gaps.begin() + 1, gaps.end());

        for (int v = 1; v <= visits; ++v) {
            const bool is_umass = uni(rng) < umass_propensity;
            const std::string facility =
                is_umass ? kUMassFacilities[static_cast<size_t>(uni(rng) * kUMassFacilities.size()) %
                                            kUMassFacilities.size()]
                         : kNonUMassFacilities[static_cast<size_t>(uni(rng) *
                                                                   kNonUMassFacilities.size()) %
                                               kNonUMassFacilities.size()];

            const DataType encounter_dt = index_dt + gaps[v - 1];
            const DataType gap = gaps[v - 1];
            const double time_fu = (v == 1)                      ? 0.0
                                   : (gap >= 1 && gap <= 30)     ? 1.0
                                   : (gap >= 31 && gap <= 91)    ? 3.0
                                   : (gap >= 92 && gap <= 182)   ? 6.0
                                                                 : 0.0;
            const double age = static_cast<double>(encounter_dt - born) / 365.0;
            const double slope = truth.beta_visit + (is_umass ? truth.slope_diff : 0.0);
            const double eta = truth.beta0 + slope * static_cast<double>(v) +
                               truth.beta_fu * time_fu + truth.beta_age * (age - 40.0) +
                               (gender == 1 ? 0.18 : 0.0) + (hispanic == 1 ? 0.10 : 0.0) +
                               u_subject;
            const bool sisa = uni(rng) < 1.0 / (1.0 + std::exp(-eta));
            const bool attempt = sisa && uni(rng) < 0.35;

            // Render the latent state as text the ETL has to parse back out.
            // suicide_acutecare_icd_narrow is ICD-only, so a SISA encounter
            // must carry an ICD code, not merely suggestive free text.
            std::string dx;
            if (sisa && attempt) {
                dx = "X781XXA INTENTIONAL SELF-HARM BY SHARP OBJECT";
            } else if (sisa) {
                dx = "R45851 SUICIDAL IDEATIONS";
            } else if (uni(rng) < 0.06) {
                // Free text only, no ICD: trips the broader text definitions
                // but not the narrow ICD outcome. This is the gap between
                // acute_care_related_sisa_pcc and suicide_acutecare_icd_narrow.
                dx = "SUICIDAL THOUGHTS REPORTED, DENIES PLAN";
            } else if (uni(rng) < 0.25) {
                dx = kDecoyDiagnoses[static_cast<size_t>(uni(rng) * kDecoyDiagnoses.size()) %
                                     kDecoyDiagnoses.size()];
            } else {
                dx = kPlainDiagnoses[static_cast<size_t>(uni(rng) * kPlainDiagnoses.size()) %
                                     kPlainDiagnoses.size()];
            }

            // A minority of encounters are neither Emergency nor Acute Care and
            // are dropped by the pass-1 filter.
            const std::string vtype = uni(rng) < 0.12  ? "Outpatient"
                                      : uni(rng) < 0.5 ? "Emergency"
                                                       : "Acute Care";

            t.PushRow(mrn, static_cast<DataType>(sid), vtype, "Hospital Encounter",
                      encounter_dt, dx, facility, born, gender, hispanic);
        }
    }
    return t;
}

// =============================================================================
// Serialisation and small helpers
// =============================================================================

// pat_mrn is a zero-padded nine-character numeric string. The conflict check
// sorts and compares it, so it crosses the boundary as an integer; the padding
// carries no information and round-trips through the width.
DataType MrnToInt(const std::string& mrn) {
    DataType v = 0;
    for (char ch : mrn)
        if (ch >= '0' && ch <= '9') v = v * 10 + (ch - '0');
    return v;
}

// Quote a CSV field that may contain commas (diagnoses_pcc and facility names
// both do).
std::string CsvQuote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += '"';
        out += c;
    }
    out += '"';
    return out;
}

void WriteBaseTableCsv(const PlainBaseTable& t, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "could not open " << path << " for writing" << std::endl;
        return;
    }
    out << "pat_mrn,subject_id,visit_type_pcc,visit_major_type_pccc,admit_date_pcc,"
           "diagnoses_pcc,visit_facility_pcc,dob,gender_pcc,hispanic\n";
    for (size_t i = 0; i < t.rows(); ++i)
        out << CsvQuote(t.pat_mrn[i]) << ',' << t.subject_id[i] << ','
            << CsvQuote(t.visit_type_pcc[i]) << ',' << CsvQuote(t.visit_major_type_pccc[i])
            << ',' << t.admit_date_pcc[i] << ',' << CsvQuote(t.diagnoses_pcc[i]) << ','
            << CsvQuote(t.visit_facility_pcc[i]) << ',' << t.dob[i] << ',' << t.gender_pcc[i]
            << ',' << t.hispanic[i] << '\n';
}

// One owner's half after pass 1. The sequencing columns are absent by
// construction: they do not exist until the two halves are merged.
void WriteFlaggedCsv(const PlainFlagged& f, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "could not open " << path << " for writing" << std::endl;
        return;
    }
    out << "pat_mrn,subject_id,encounter_dt,newage,gender,hispanic,data_source,"
           "suicide_acutecare_icd_narrow,suicide_acutecare_icd_broad,sa_icd_narrow,"
           "acute_care_related_sisa\n";
    for (size_t i = 0; i < f.rows(); ++i)
        out << CsvQuote(f.pat_mrn[i]) << ',' << f.subject_id[i] << ',' << f.encounter_dt[i]
            << ',' << f.newage[i] << ',' << f.gender[i] << ',' << f.hispanic[i] << ','
            << f.data_source[i] << ',' << f.suicide_acutecare_icd_narrow[i] << ','
            << f.suicide_acutecare_icd_broad[i] << ',' << f.sa_icd_narrow_pcc[i] << ','
            << f.acute_care_related_sisa_pcc[i] << '\n';
}

// =============================================================================
// Reading a base table back
//
// The import path for a real deployment. Each owner holds its OWN half of
// cdrcatsse_match_pcc as a CSV and opens only that file; pass 1 then runs
// locally over it, exactly as in the synthetic path. No party ever reads the
// other's file and no plaintext crosses the wire.
//
// This replaces the three-table `any_system.csv` / `umass_system.csv` /
// `nonumass_system.csv` import removed in task 0011. That import assumed one
// party held each finished analysis table in the clear, which is precisely the
// assumption this pipeline exists to drop: the sequencing columns those files
// carried cannot be computed by any single owner.
// =============================================================================

const std::vector<std::string> kBaseCsvHeader = {
    "pat_mrn",       "subject_id",         "visit_type_pcc", "visit_major_type_pccc",
    "admit_date_pcc", "diagnoses_pcc",     "visit_facility_pcc", "dob",
    "gender_pcc",    "hispanic"};

// Split one CSV line, honouring double-quoted fields. diagnoses_pcc and the
// facility names both contain commas, so a naive split corrupts every row after
// the first quoted one.
std::vector<std::string> SplitCsvLine(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (quoted) {
            if (ch == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cur += '"';
                    ++i;
                } else {
                    quoted = false;
                }
            } else {
                cur += ch;
            }
        } else if (ch == '"') {
            quoted = true;
        } else if (ch == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += ch;
        }
    }
    out.push_back(cur);
    return out;
}

// Columns are matched by NAME, as EncodedTable::inputCSVTableData does, so the
// order in the file is free -- but every expected name must be present.
PlainBaseTable ReadBaseTableCsv(const std::string& path) {
    PlainBaseTable t;
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
    std::vector<std::string> header = SplitCsvLine(line);
    for (std::string& h : header)
        while (!h.empty() && (h.back() == '\r' || h.back() == ' ')) h.pop_back();

    std::map<std::string, size_t> pos;
    for (size_t i = 0; i < header.size(); ++i) pos[header[i]] = i;
    for (const std::string& want : kBaseCsvHeader) {
        if (pos.find(want) == pos.end()) {
            std::cerr << "FATAL: column " << want << " missing from " << path << std::endl;
            std::exit(1);
        }
    }

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const std::vector<std::string> f = SplitCsvLine(line);
        auto str = [&](const char* name) -> std::string {
            const size_t k = pos[name];
            return k < f.size() ? f[k] : std::string();
        };
        auto num = [&](const char* name) -> DataType {
            const std::string v = str(name);
            return v.empty() ? DataType{0} : static_cast<DataType>(std::stoll(v));
        };
        t.PushRow(str("pat_mrn"), num("subject_id"), str("visit_type_pcc"),
                  str("visit_major_type_pccc"), num("admit_date_pcc"), str("diagnoses_pcc"),
                  str("visit_facility_pcc"), num("dob"), num("gender_pcc"), num("hispanic"));
    }
    return t;
}

// One owner's half for a cross-organisational run. Every party calls this; only
// the owner opens the file.
//
// A non-owner gets an EMPTY table, not a zero-filled one of the right length.
// That is the difference from the removed LoadOwnedCohort: sharing used to be
// per-table and needed a matching row count locally, whereas MergeTwoOwners
// sizes both halves from the public manifest counts and shares each owner's
// vector independently, so the non-owner never needs the shape.
//
// `simulate_all` is for a build where ONE process stands in for every party --
// the 1PC protocol, where getPartyID() is always 0. There the ownership guard
// would leave owner B's half permanently empty and the merge would silently see
// half the data. A local simulation reading both files discloses nothing,
// because there is only one party to disclose to.
PlainBaseTable LoadOwnedBaseTable(const std::string& data_dir, const std::string& file,
                                  int owner, int party_id, bool simulate_all) {
    if (!simulate_all && party_id != owner) return PlainBaseTable{};
    return ReadBaseTableCsv(data_dir + "/" + file);
}

// =============================================================================
// Simulating two data owners
//
// The lineage has one base table. We split it to model the deployment: two
// organisations each holding part of the encounter history, overlapping on
// subject_id because a patient can be seen at both.
//
// The split is by ENCOUNTER, not by patient -- that is the whole point. A
// patient's rows land at both owners, so no owner can sequence that patient
// alone. `overlap` sets how strongly a patient's encounters concentrate at one
// owner: 1.0 sends every patient entirely to one side (no overlap, and the
// merge is unnecessary), 0.5 splits each patient's encounters evenly.
//
// A patient is also given a small chance of carrying a DIFFERENT study ID at
// the second owner, which is what conflict_list exists to detect and what no
// single owner can see.
// =============================================================================

struct OwnerSplit {
    PlainBaseTable a;
    PlainBaseTable b;
};

OwnerSplit SplitAcrossOwners(const PlainBaseTable& t, double concentration = 0.7,
                             double id_conflict_rate = 0.0, unsigned seed = 7) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    OwnerSplit s;
    std::map<DataType, bool> home;       // which owner this patient mostly uses
    std::map<DataType, DataType> alias;  // a second study ID at owner B

    for (size_t i = 0; i < t.rows(); ++i) {
        const DataType sid = t.subject_id[i];
        if (home.find(sid) == home.end()) {
            home[sid] = uni(rng) < 0.5;
            if (uni(rng) < id_conflict_rate) alias[sid] = sid + 100000;
        }
        // Mostly the patient's home owner, sometimes the other one.
        const bool at_home = uni(rng) < concentration;
        const bool to_a = at_home ? home[sid] : !home[sid];

        PlainBaseTable& dst = to_a ? s.a : s.b;
        const DataType sid_here =
            (!to_a && alias.count(sid)) ? alias[sid] : sid;
        dst.PushRow(t.pat_mrn[i], sid_here, t.visit_type_pcc[i], t.visit_major_type_pccc[i],
                    t.admit_date_pcc[i], t.diagnoses_pcc[i], t.visit_facility_pcc[i], t.dob[i],
                    t.gender_pcc[i], t.hispanic[i]);
    }
    return s;
}

}  // namespace cdough::regression

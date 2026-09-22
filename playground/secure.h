#pragma once

#include <cassert>
#include <string>
#include <vector>

#include "./cohort.h"
#include "./etl.h"
#include "./segmented.h"

// The cross-party half of the pipeline: everything that cannot be computed by
// one data owner over its own rows.
//
// pcc_mrn and flagged_dx pass 1 are row-local and run at each
// owner in the clear, which keeps every LIKE '%...%' out of MPC. What is left
// partitions by subject_id or pat_mrn, and a patient's encounters live at BOTH
// owners, so those nodes run here:
//
//   conflict_list      pat_mrn -> COUNT(DISTINCT subject_id) > 1
//   flagged_dx pass 2  ROW_NUMBER / FIRST_VALUE over (PARTITION BY subject_id
//                      ORDER BY encounter_dt)
//   umass / nonumass   the same windows, re-sequenced per system
//
// Order of operations:
//
//   share each owner's locally-sorted half into the two halves of one N-row
//   table, N = 2m
//              |
//   bitonic_merge on {subject_id, encounter_dt}        log N stages
//              |
//   pass 2: global sequencing on {subject_id}          O(log N) rounds
//              |
//   bitonic_sort on {subject_id, data_source, encounter_dt}
//              |
//   per-system sequencing on {subject_id, data_source} O(log N) rounds
//
// The second ordering is needed because {subject_id, data_source} runs are NOT
// contiguous after a date sort -- a patient's encounters interleave between
// systems -- and the segmented scans require adjacency. Sorting on
// encounter_dt as the third key reproduces date order inside each
// (patient, system) run without needing a stable sort; bitonic sort is not
// stable.

namespace cdough::regression {

using cdough::operators::SortOrder;

// =============================================================================
// Column plumbing
// =============================================================================

// One owner's share of the merged table, already reduced to the columns the
// analysis needs. Keys are B-shared because they are sorted and compared;
// values are A-shared because they are summed.
struct SecureColumns {
    size_t n_pad = 0;

    BV subject_key;  // sort key 1 and the grouping key
    BV dt_key;       // sort key 2: encounter_dt
    BV ds_key;       // sort key 3 for the per-system pass: data_source

    AV valid;         // 1 on real rows, 0 on pad
    AV encounter_dt;  // an arithmetic copy of dt_key, for the date arithmetic
    AV newage;
    AV sisa;  // suicide_acutecare_icd_narrow, the outcome
    AV sa;    // sa_icd_narrow
    AV umass;  // 1 where data_source == 1
    std::vector<AV> gender_is;
    std::vector<AV> hispanic_is;

    SecureColumns(EngineRef engine, size_t padded)
        : n_pad(padded),
          subject_key(padded, engine),
          dt_key(padded, engine),
          ds_key(padded, engine),
          valid(padded, engine),
          encounter_dt(padded, engine),
          newage(padded, engine),
          sisa(padded, engine),
          sa(padded, engine),
          umass(padded, engine) {}

    // Every A-shared column, in one list, for handing to the sort as payload.
    std::vector<AV*> DataA() {
        std::vector<AV*> out{&valid, &encounter_dt, &newage, &sisa, &sa, &umass};
        for (AV& g : gender_is) out.push_back(&g);
        for (AV& h : hispanic_is) out.push_back(&h);
        return out;
    }
};

// Build the plaintext column vectors one owner contributes, padded to `m`.
//
// The owner has already sorted its half by (subject_id, encounter_dt) in the
// clear -- free, and it reveals nothing the owner does not hold. Pad rows carry
// the key sentinel in BOTH key columns so they sort last under either and form
// one contiguous block, and zero everywhere else so they contribute nothing.
struct HalfPlain {
    cdough::Vector<DataType> subject, dt, ds, valid, newage, sisa, sa, umass;
    std::vector<cdough::Vector<DataType>> gender_is, hispanic_is;

    HalfPlain(size_t m)
        : subject(m, 0), dt(m, 0), ds(m, 0), valid(m, 0), newage(m, 0), sisa(m, 0), sa(m, 0),
          umass(m, 0) {}
};

HalfPlain BuildHalf(const PlainFlagged& f, const std::vector<size_t>& order, size_t m) {
    HalfPlain h(m);
    for (size_t lvl = 0; lvl < kGenderLevels.size(); ++lvl)
        h.gender_is.emplace_back(m, 0);
    for (size_t lvl = 0; lvl < kHispanicLevels.size(); ++lvl)
        h.hispanic_is.emplace_back(m, 0);

    for (size_t i = 0; i < m; ++i) {
        if (i < order.size()) {
            const size_t r = order[i];
            h.subject[i] = f.subject_id[r];
            h.dt[i] = f.encounter_dt[r];
            h.ds[i] = f.data_source[r];
            h.valid[i] = 1;
            h.newage[i] = f.newage[r] * DataType(scale);
            h.sisa[i] = f.suicide_acutecare_icd_narrow[r];
            h.sa[i] = f.sa_icd_narrow_pcc[r];
            h.umass[i] = (f.data_source[r] == 1) ? 1 : 0;
            for (size_t k = 0; k < kGenderLevels.size(); ++k)
                h.gender_is[k][i] = (f.gender[r] == kGenderLevels[k]) ? 1 : 0;
            for (size_t k = 0; k < kHispanicLevels.size(); ++k)
                h.hispanic_is[k][i] = (f.hispanic[r] == kHispanicLevels[k]) ? 1 : 0;
        } else {
            h.subject[i] = kKeySentinel;
            h.dt[i] = kKeySentinel;
            h.ds[i] = kKeySentinel;
        }
    }
    return h;
}

// The owner's local sort: (subject_id, encounter_dt) ascending.
std::vector<size_t> LocalOrder(const PlainFlagged& f) {
    std::vector<size_t> order(f.rows());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (f.subject_id[a] != f.subject_id[b]) return f.subject_id[a] < f.subject_id[b];
        return f.encounter_dt[a] < f.encounter_dt[b];
    });
    return order;
}

// =============================================================================
// Ingestion and the oblivious merge
//
// Both halves pad to the same m, so the concatenation is 2m and the two halves
// are equal -- bitonic_merge requires that, and it also means the row counts do
// not leak which owner is larger beyond what the manifest already publishes.
// =============================================================================

SecureColumns MergeTwoOwners(EngineRef engine, const PlainFlagged& half_a, int party_a,
                             const PlainFlagged& half_b, int party_b, size_t rows_a,
                             size_t rows_b) {
    const size_t m = NextPowerOfTwo(std::max<size_t>(std::max(rows_a, rows_b), 2));
    const size_t n = 2 * m;

    HalfPlain pa = BuildHalf(half_a, LocalOrder(half_a), m);
    HalfPlain pb = BuildHalf(half_b, LocalOrder(half_b), m);

    SecureColumns c(engine, n);
    for (size_t k = 0; k < kGenderLevels.size(); ++k) c.gender_is.emplace_back(n, engine);
    for (size_t k = 0; k < kHispanicLevels.size(); ++k) c.hispanic_is.emplace_back(n, engine);

    // Each owner shares its own half; only the owning party's vector carries
    // real values, and every other party passes an empty placeholder.
    auto share_b_half = [&](BV& dst, cdough::Vector<DataType>& va,
                            cdough::Vector<DataType>& vb) {
        BV sa = engine.template secret_share_b<DataType>(va, party_a);
        BV sb = engine.template secret_share_b<DataType>(vb, party_b);
        BV lo = dst.slice(0, m);
        lo = sa;
        BV hi = dst.slice(m, n);
        hi = sb;
    };
    auto share_a_half = [&](AV& dst, cdough::Vector<DataType>& va,
                            cdough::Vector<DataType>& vb) {
        AV sa = engine.template secret_share_a<DataType>(va, party_a, 0);
        AV sb = engine.template secret_share_a<DataType>(vb, party_b, 0);
        AV lo = dst.slice(0, m);
        lo = sa;
        AV hi = dst.slice(m, n);
        hi = sb;
        dst.setPrecision(0);
    };

    share_b_half(c.subject_key, pa.subject, pb.subject);
    share_b_half(c.dt_key, pa.dt, pb.dt);
    share_b_half(c.ds_key, pa.ds, pb.ds);
    share_a_half(c.valid, pa.valid, pb.valid);
    share_a_half(c.encounter_dt, pa.dt, pb.dt);
    share_a_half(c.newage, pa.newage, pb.newage);
    share_a_half(c.sisa, pa.sisa, pb.sisa);
    share_a_half(c.sa, pa.sa, pb.sa);
    share_a_half(c.umass, pa.umass, pb.umass);
    for (size_t k = 0; k < kGenderLevels.size(); ++k)
        share_a_half(c.gender_is[k], pa.gender_is[k], pb.gender_is[k]);
    for (size_t k = 0; k < kHispanicLevels.size(); ++k)
        share_a_half(c.hispanic_is[k], pa.hispanic_is[k], pb.hispanic_is[k]);

    // The merge itself. Each half is already ascending on (subject, dt), which
    // is exactly bitonic_merge's contract, so this costs log N compare-swap
    // stages rather than the log N (log N + 1) / 2 a full sort would.
    //
    // NOTE: the by-value overload of bitonic_merge (merge.h:245) pushes the key
    // columns into the result three times and drops every data payload. Use the
    // pointer overload.
    std::vector<BV*> keys{&c.subject_key, &c.dt_key};
    std::vector<BV*> data_b{&c.ds_key};
    std::vector<AV*> data_a = c.DataA();
    cdough::operators::bitonic_merge(keys, data_a, data_b,
                                     {SortOrder::ASC, SortOrder::ASC});
    return c;
}

// =============================================================================
// Single-owner ingestion
//
// The other data model the deployment supports: ONE organisation holds the whole
// of cdrcatsse_match_pcc and the other parties hold nothing. It is the shape a
// single-site run takes, and the shape `--owner` selects.
//
// It is not the two-owner path with an empty second half. Two things fall away:
//
//   - There is no merge. bitonic_merge exists to interleave two independently
//     sorted halves; one owner sorts its whole table in the clear, so the shared
//     table arrives already ordered on (subject_id, encounter_dt) and the log N
//     compare-swap stages have nothing left to do.
//   - The table is padded to NextPowerOfTwo(rows) rather than 2 *
//     NextPowerOfTwo(rows), because there is no second half to match. That halves
//     the row count every downstream sort and scan runs over.
//
// The local sort discloses nothing: the owner is sorting rows it already holds
// in plaintext. Everything after this point is identical to the two-owner path,
// which is why RunSecurePipeline and RunSecurePipelineSingleOwner share
// SequenceSharedTable below rather than each carrying their own copy of it.
// =============================================================================

SecureColumns ShareOneOwner(EngineRef engine, const PlainFlagged& half, int party,
                            size_t rows) {
    const size_t n = NextPowerOfTwo(std::max<size_t>(rows, 2));

    HalfPlain p = BuildHalf(half, LocalOrder(half), n);

    SecureColumns c(engine, n);
    for (size_t k = 0; k < kGenderLevels.size(); ++k) c.gender_is.emplace_back(n, engine);
    for (size_t k = 0; k < kHispanicLevels.size(); ++k) c.hispanic_is.emplace_back(n, engine);

    // Every party calls these; only `party` holds real values, so the others pass
    // the zero-filled placeholder BuildHalf gave them and receive shares.
    auto share_b = [&](BV& dst, cdough::Vector<DataType>& v) {
        dst = engine.template secret_share_b<DataType>(v, party);
    };
    auto share_a = [&](AV& dst, cdough::Vector<DataType>& v) {
        dst = engine.template secret_share_a<DataType>(v, party, 0);
        dst.setPrecision(0);
    };

    share_b(c.subject_key, p.subject);
    share_b(c.dt_key, p.dt);
    share_b(c.ds_key, p.ds);
    share_a(c.valid, p.valid);
    share_a(c.encounter_dt, p.dt);
    share_a(c.newage, p.newage);
    share_a(c.sisa, p.sisa);
    share_a(c.sa, p.sa);
    share_a(c.umass, p.umass);
    for (size_t k = 0; k < kGenderLevels.size(); ++k) share_a(c.gender_is[k], p.gender_is[k]);
    for (size_t k = 0; k < kHispanicLevels.size(); ++k)
        share_a(c.hispanic_is[k], p.hispanic_is[k]);

    // No bitonic_merge: LocalOrder already left the table ascending on
    // (subject_id, encounter_dt), which is what the merge exists to produce.
    return c;
}

// Re-order the merged table by (subject_id, data_source, encounter_dt) so that
// each (patient, system) run is contiguous and internally in date order. The
// already-computed global sequencing columns ride along as payload.
void SortBySystem(SecureColumns& c, std::vector<AV*> extra) {
    std::vector<BV*> keys{&c.subject_key, &c.ds_key, &c.dt_key};
    std::vector<BV*> data_b;
    std::vector<AV*> data_a = c.DataA();
    for (AV* e : extra) data_a.push_back(e);
    cdough::operators::bitonic_sort(keys, data_a, data_b,
                                    {SortOrder::ASC, SortOrder::ASC, SortOrder::ASC});
}

// =============================================================================
// flagged_dx pass 2 -- the sequencing
//
// The five window functions of the upstream query, each mapped onto a segmented
// primitive. The table must already be sorted by (key, encounter_dt).
// =============================================================================

struct Sequencing {
    AV visit_num;    // ROW_NUMBER() OVER w
    AV index_visit;  // ROW_NUMBER() OVER w = 1
    AV final_visit;  // ROW_NUMBER() OVER w_desc = 1
    AV index_dt;     // FIRST_VALUE(encounter_dt) OVER w
    AV fu_month;     // the follow-up banding
    AV fu_present;   // 0 encodes SQL NULL
};

// fu_month: a banding of the gap from the patient's index encounter.
//
//   0 on the index visit;  1 for 1..30 days;  3 for 31..91;  6 for 92..182;
//   NULL otherwise -- an encounter at 183 days or more, and also a same-day
//   repeat, because the 1-month band starts strictly above a zero-day gap.
//
// Guarded on the index_visit of THIS scope. The upstream ETL guards the
// per-system bands on the GLOBAL index_visit instead, which is the documented
// fault that nulls fu_month on the first in-system encounter of any patient
// whose in-system history starts later than their overall history. Computing it
// here makes the correct flag the natural one to use, so the MPC pipeline does
// not reproduce that bug; the plaintext oracle matches this behaviour.
void BandFollowUp(const AV& gap, const AV& index_visit, AV& fu_month, AV& fu_present) {
    auto in_range = [&](DataType lo, DataType hi) {
        AV above = Clone(gap);
        above.setPrecision(0);
        above -= lo;
        AV ge = *(above.gtez());  // gap >= lo
        AV below = Clone(gap);
        below.setPrecision(0);
        AV le_src = *(below * DataType(-1));
        le_src.setPrecision(0);
        le_src += hi;
        AV le = *(le_src.gtez());  // gap <= hi
        AV both = *(ge * le);
        both.setPrecision(0);
        return both;
    };

    AV in1 = in_range(1, 30);
    AV in3 = in_range(31, 91);
    AV in6 = in_range(92, 182);

    // Bands only apply off the index visit; the index visit is 0 by definition.
    AV not_index = Clone(index_visit);
    not_index.setPrecision(0);
    AV one_minus = *(not_index * DataType(-1));
    one_minus.setPrecision(0);
    one_minus += DataType(1);

    AV b1 = *(in1 * one_minus);
    b1.setPrecision(0);
    AV b3 = *(in3 * one_minus);
    b3.setPrecision(0);
    AV b6 = *(in6 * one_minus);
    b6.setPrecision(0);

    AV fu = *(b1 * DataType(1));
    fu.setPrecision(0);
    AV t3 = *(b3 * DataType(3));
    t3.setPrecision(0);
    AV t6 = *(b6 * DataType(6));
    t6.setPrecision(0);
    fu += t3;
    fu += t6;
    fu.setPrecision(0);

    // present = index_visit OR in any band. The bands are disjoint and each is
    // already ANDed with "not the index visit", so a sum is an OR here.
    AV present = Clone(index_visit);
    present.setPrecision(0);
    present += b1;
    present += b3;
    present += b6;
    present.setPrecision(0);

    fu_month = fu;
    fu_present = present;
}

// Compute the sequencing over `keys`, which must already group the table.
Sequencing SequenceVisits(std::vector<BV>& keys, const ScanPlan& plan, const AV& valid,
                          const AV& encounter_dt) {
    Sequencing s{Clone(valid), Clone(valid), Clone(valid),
                 Clone(valid), Clone(valid), Clone(valid)};

    // ROW_NUMBER() OVER w -- the rank. A segmented prefix sum over the ones in
    // `valid`, evaluated on the Brent-Kung network: O(log n) rounds.
    s.visit_num = SegRankPlanned(plan, valid, SegDirection::Forward);

    // index_visit / final_visit. The AND with `valid` is what stops the
    // sentinel pad block contributing a spurious group.
    BV first_b = FirstOfGroup(keys);
    AV first = *(first_b.b2a_bit());
    first.setPrecision(0);
    s.index_visit = *(first * valid);
    s.index_visit.setPrecision(0);

    AV last = LastOfGroupArith(keys);
    s.final_visit = *(last * valid);
    s.final_visit.setPrecision(0);

    // FIRST_VALUE(encounter_dt) OVER w
    s.index_dt = SegFirstValuePlanned(plan, encounter_dt, s.index_visit);

    AV gap = Clone(encounter_dt);
    gap.setPrecision(0);
    gap -= s.index_dt;
    gap.setPrecision(0);
    BandFollowUp(gap, s.index_visit, s.fu_month, s.fu_present);

    // Pad rows must stay inert in every derived column.
    s.visit_num = *(s.visit_num * valid);
    s.visit_num.setPrecision(0);
    s.fu_month = *(s.fu_month * valid);
    s.fu_month.setPrecision(0);
    s.fu_present = *(s.fu_present * valid);
    s.fu_present.setPrecision(0);
    return s;
}

// =============================================================================
// conflict_list
//
// pat_mrn -> COUNT(DISTINCT subject_id), keeping the MRNs above 1.
//
// Upstream this is a dead end that nothing reads. Here it stops being one: an
// MRN can carry one study ID at each owner, and neither owner can see the
// disagreement alone. It needs its own ordering, because the analysis table is
// sorted by subject and MRN groups are not contiguous in it.
//
// Only the NUMBER of conflicting MRNs is opened. That is the count the upstream
// table publishes, and it discloses nothing about which patients are involved.
//
// `reveal_to` names the party that learns that count; -1 gives it to everybody,
// which is the two-owner default. Collective either way.
// =============================================================================

long SecureConflictCount(EngineRef engine, const std::vector<DataType>& mrn_a,
                         const std::vector<DataType>& sid_a, int party_a,
                         const std::vector<DataType>& mrn_b,
                         const std::vector<DataType>& sid_b, int party_b,
                         int reveal_to = -1, int party_id = 0) {
    const size_t m = NextPowerOfTwo(std::max<size_t>(std::max(mrn_a.size(), mrn_b.size()), 2));
    const size_t n = 2 * m;

    auto fill = [&](const std::vector<DataType>& v, size_t take) {
        cdough::Vector<DataType> out(m, 0);
        for (size_t i = 0; i < m; ++i) out[i] = i < take ? v[i] : kKeySentinel;
        return out;
    };
    auto fill_valid = [&](size_t take) {
        cdough::Vector<DataType> out(m, 0);
        for (size_t i = 0; i < m; ++i) out[i] = i < take ? 1 : 0;
        return out;
    };

    cdough::Vector<DataType> ma = fill(mrn_a, mrn_a.size()), sa_ = fill(sid_a, sid_a.size());
    cdough::Vector<DataType> mb = fill(mrn_b, mrn_b.size()), sb_ = fill(sid_b, sid_b.size());
    cdough::Vector<DataType> va = fill_valid(mrn_a.size()), vb = fill_valid(mrn_b.size());

    BV mrn(n, engine), sid(n, engine);
    AV valid(n, engine);
    {
        BV x = engine.template secret_share_b<DataType>(ma, party_a);
        BV y = engine.template secret_share_b<DataType>(mb, party_b);
        BV lo = mrn.slice(0, m); lo = x;
        BV hi = mrn.slice(m, n); hi = y;
    }
    {
        BV x = engine.template secret_share_b<DataType>(sa_, party_a);
        BV y = engine.template secret_share_b<DataType>(sb_, party_b);
        BV lo = sid.slice(0, m); lo = x;
        BV hi = sid.slice(m, n); hi = y;
    }
    {
        AV x = engine.template secret_share_a<DataType>(va, party_a, 0);
        AV y = engine.template secret_share_a<DataType>(vb, party_b, 0);
        AV lo = valid.slice(0, m); lo = x;
        AV hi = valid.slice(m, n); hi = y;
        valid.setPrecision(0);
    }

    // Order by (pat_mrn, subject_id) so both the MRN runs and the distinct
    // study IDs inside them are contiguous.
    std::vector<BV*> keys{&mrn, &sid};
    std::vector<AV*> data_a{&valid};
    std::vector<BV*> data_b;
    cdough::operators::bitonic_sort(keys, data_a, data_b,
                                    {SortOrder::ASC, SortOrder::ASC});

    // A row starting a new (mrn, subject_id) run is a distinct study ID for its
    // MRN. Summing those per MRN gives COUNT(DISTINCT subject_id).
    std::vector<BV> pair_keys{mrn, sid};
    BV new_pair_b = FirstOfGroup(pair_keys);
    AV new_pair = *(new_pair_b.b2a_bit());
    new_pair.setPrecision(0);
    new_pair = *(new_pair * valid);
    new_pair.setPrecision(0);

    std::vector<BV> mrn_keys{mrn};
    AV per_mrn = SegTotal(mrn_keys, new_pair);
    per_mrn.setPrecision(0);

    // Count each MRN once, on its first row, and only when the count exceeds 1.
    BV first_mrn_b = FirstOfGroup(mrn_keys);
    AV first_mrn = *(first_mrn_b.b2a_bit());
    first_mrn.setPrecision(0);
    first_mrn = *(first_mrn * valid);
    first_mrn.setPrecision(0);

    AV over = Clone(per_mrn);
    over.setPrecision(0);
    over -= DataType(2);
    AV conflicted = *(over.gtez());  // count >= 2
    conflicted.setPrecision(0);
    AV hit = *(conflicted * first_mrn);
    hit.setPrecision(0);

    AV total = hit.chunkedSum(hit.size());
    total.setPrecision(0);
    return static_cast<long>(
        std::llround(OpenScalarToParty(total, reveal_to, party_id, false)));
}

// =============================================================================
// Assembling the three analysis cohorts
//
// All three share one row set. They differ only in which rows their `valid`
// mask admits and which sequencing columns they carry, which is what lets the
// per-system re-sequencing come out of a single extra pass instead of two.
// =============================================================================

SecureCohort BuildCohort(EngineRef engine, SecureColumns& c, const Sequencing& seq,
                         const AV& scope_valid, const std::vector<BV>& keys,
                         const ScanPlan& plan, SystemScope scope) {
    SecureCohort sc(engine, c.n_pad, c.n_pad);
    sc.scope = scope;
    sc.subject_key = c.subject_key;
    sc.keys = keys;
    sc.scan_plan = plan;

    sc.valid = Clone(scope_valid);
    sc.valid.setPrecision(0);

    auto masked = [&](const AV& col) {
        AV out = *(col * sc.valid);
        out.setPrecision(0);
        return out;
    };

    sc.newage = Clone(c.newage);
    sc.newage.setPrecision(0);
    sc.encounter_dt = Clone(c.encounter_dt);
    sc.encounter_dt.setPrecision(0);
    sc.visit_num = *(seq.visit_num * DataType(scale));
    sc.visit_num.setPrecision(0);
    sc.fu_month = *(seq.fu_month * DataType(scale));
    sc.fu_month.setPrecision(0);
    sc.fu_present = masked(seq.fu_present);
    sc.index_visit = masked(seq.index_visit);
    sc.final_visit = masked(seq.final_visit);
    sc.sisa = Clone(c.sisa);
    sc.sisa.setPrecision(0);
    sc.sa = Clone(c.sa);
    sc.sa.setPrecision(0);
    sc.umass = Clone(c.umass);
    sc.umass.setPrecision(0);
    for (AV& g : c.gender_is) sc.gender_is.push_back(Clone(g));
    for (AV& h : c.hispanic_is) sc.hispanic_is.push_back(Clone(h));

    std::vector<BV> k = keys;
    AV last = LastOfGroupArith(k);
    sc.last_of_subject = *(last * sc.valid);
    sc.last_of_subject.setPrecision(0);

    BV first_b = FirstOfGroup(k);
    AV first = *(first_b.b2a_bit());
    first.setPrecision(0);
    sc.first_of_subject = *(first * sc.valid);
    sc.first_of_subject.setPrecision(0);

    AV first_count = sc.first_of_subject.chunkedSum(c.n_pad);
    first_count.setPrecision(0);
    sc.num_subjects = static_cast<size_t>(std::llround(OpenScalar(first_count, false)));
    return sc;
}

// =============================================================================
// The plaintext oracle for the cross-party half
//
// The same nodes -- union, sequence, re-sequence per system -- computed in the
// clear over both owners' rows at once. Nothing in the deployment may run this:
// it needs the union, which is exactly what no party holds. It exists so the
// MPC pipeline has something to be checked against, and it is also what the
// regression oracle in reporting.h scores the fits on.
//
// `scope` selects which rows participate BEFORE sequencing, which is what makes
// this the per-system re-sequencing for UMass / NonUMass: a patient's third
// encounter overall may be their first inside one system.
// =============================================================================

PlainCohort SequencePlain(const PlainFlagged& a, const PlainFlagged& b, SystemScope scope) {
    struct Row {
        DataType sid, dt, ds, newage, gender, hispanic, sisa, sa;
    };
    std::vector<Row> rows;
    auto take = [&](const PlainFlagged& f) {
        for (size_t i = 0; i < f.rows(); ++i) {
            if (scope == SystemScope::UMass && f.data_source[i] != 1) continue;
            if (scope == SystemScope::NonUMass && f.data_source[i] != 2) continue;
            rows.push_back(Row{f.subject_id[i], f.encounter_dt[i], f.data_source[i],
                               f.newage[i], f.gender[i], f.hispanic[i],
                               f.suicide_acutecare_icd_narrow[i], f.sa_icd_narrow_pcc[i]});
        }
    };
    take(a);
    take(b);

    std::stable_sort(rows.begin(), rows.end(), [](const Row& x, const Row& y) {
        if (x.sid != y.sid) return x.sid < y.sid;
        return x.dt < y.dt;
    });

    PlainCohort c;
    c.scope = scope;
    size_t i = 0;
    while (i < rows.size()) {
        size_t j = i;
        while (j < rows.size() && rows[j].sid == rows[i].sid) ++j;
        const DataType index_dt = rows[i].dt;
        for (size_t k = i; k < j; ++k) {
            const int v = static_cast<int>(k - i) + 1;
            const DataType gap = rows[k].dt - index_dt;
            DataType fu = 0, fu_present = 1;
            if (v == 1) fu = 0;
            else if (gap >= 1 && gap <= 30) fu = 1;
            else if (gap >= 31 && gap <= 91) fu = 3;
            else if (gap >= 92 && gap <= 182) fu = 6;
            else fu_present = 0;

            c.subject_id.push_back(rows[k].sid);
            c.encounter_dt.push_back(rows[k].dt);
            c.newage.push_back(rows[k].newage);
            c.gender.push_back(rows[k].gender);
            c.hispanic.push_back(rows[k].hispanic);
            c.index_visit.push_back(v == 1 ? 1 : 0);
            c.visit_num.push_back(v);
            c.final_visit.push_back(k + 1 == j ? 1 : 0);
            c.fu_month.push_back(fu);
            c.fu_month_present.push_back(fu_present);
            c.data_source.push_back(rows[k].ds);
            c.sisa.push_back(rows[k].sisa);
            c.sa.push_back(rows[k].sa);
        }
        i = j;
    }
    return c;
}

// =============================================================================
// The whole cross-party pipeline
//
//   merge -> global sequencing -> re-order by system -> per-system sequencing
//            -> the three analysis cohorts
//
// Both sequencing passes call the same SequenceVisits; they differ only in the
// grouping key, which is what makes the per-system re-sequencing of the umass /
// nonumass lineage nodes fall out of one extra ordering rather than two more
// pipelines.
// =============================================================================

struct SecurePipeline {
    SecureCohort any;
    SecureCohort umass;
    SecureCohort nonumass;
};

// Everything after ingestion. Both data models -- two owners merged, or one
// owner's whole table -- reach this with the same object: a shared table already
// ascending on (subject_id, encounter_dt). Nothing below can tell which produced
// it, which is the point of the split.
//
// `n_real` is the number of REAL rows, for the cohorts' reported row counts.
SecurePipeline SequenceSharedTable(EngineRef engine, SecureColumns& c, size_t n_real) {
    // --- flagged_dx pass 2: sequencing on the date order ---------------------
    std::vector<BV> subject_keys{c.subject_key};
    ScanPlan plan_dt = BuildScanPlan(c.subject_key);
    Sequencing global = SequenceVisits(subject_keys, plan_dt, c.valid, c.encounter_dt);

    // --- re-order so (patient, system) runs are contiguous -------------------
    // The global columns ride along as payload; recomputing them after the sort
    // would be wrong, because this order is system-then-date, not date.
    std::vector<AV*> carry{&global.visit_num, &global.index_visit, &global.final_visit,
                           &global.index_dt,  &global.fu_month,    &global.fu_present};
    SortBySystem(c, carry);

    // --- umass / nonumass: the same windows on the compound key --------------
    std::vector<BV> sys_keys{c.subject_key, c.ds_key};
    ScanPlan plan_sys = BuildScanPlan(sys_keys);
    Sequencing per_system = SequenceVisits(sys_keys, plan_sys, c.valid, c.encounter_dt);

    // The subject runs are still contiguous after the re-order, since
    // subject_id is the leading sort key -- so the pooled cohort groups
    // correctly on this row order too. Its plan has to be rebuilt, though: the
    // rows moved.
    ScanPlan plan_any = BuildScanPlan(c.subject_key);

    AV umass_valid = *(c.valid * c.umass);
    umass_valid.setPrecision(0);
    AV not_umass = *(c.umass * DataType(-1));
    not_umass.setPrecision(0);
    not_umass += DataType(1);
    AV nonumass_valid = *(c.valid * not_umass);
    nonumass_valid.setPrecision(0);

    SecurePipeline out{
        BuildCohort(engine, c, global, c.valid, subject_keys, plan_any, SystemScope::Any),
        BuildCohort(engine, c, per_system, umass_valid, sys_keys, plan_sys, SystemScope::UMass),
        BuildCohort(engine, c, per_system, nonumass_valid, sys_keys, plan_sys,
                    SystemScope::NonUMass)};
    out.any.n = n_real;
    out.umass.n = n_real;
    out.nonumass.n = n_real;
    return out;
}

// `rows_a` / `rows_b` are PUBLIC row counts from the run manifest, not
// half_a.rows() / half_b.rows(). In a cross-organisational run a party holds
// only its own half, so the other one is empty locally -- deriving the padded
// size from the local vectors would give each party a different N and the
// shares would not line up.
SecurePipeline RunSecurePipeline(EngineRef engine, const PlainFlagged& half_a, int party_a,
                                 const PlainFlagged& half_b, int party_b, size_t rows_a,
                                 size_t rows_b) {
    SecureColumns c = MergeTwoOwners(engine, half_a, party_a, half_b, party_b, rows_a, rows_b);
    return SequenceSharedTable(engine, c, rows_a + rows_b);
}

// The single-owner data model. `rows` is the manifest row count after the pass-1
// filter, public for the same reason rows_a / rows_b are: every party pads from
// it, and the parties that do not own the file cannot read its length.
SecurePipeline RunSecurePipelineSingleOwner(EngineRef engine, const PlainFlagged& half,
                                            int party, size_t rows) {
    SecureColumns c = ShareOneOwner(engine, half, party, rows);
    return SequenceSharedTable(engine, c, rows);
}

}  // namespace cdough::regression

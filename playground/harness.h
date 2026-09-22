#pragma once

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "./regression.h"
#include "./secure.h"
#include "./sqlite_oracle.h"  // no-op unless HAVE_SQLITE3

// Accuracy harnesses and the cost breakdown. Every kernel added by this
// pipeline is checked against a plaintext oracle here before anything is
// built on top of it.

namespace cdough::regression {

// =============================================================================
// Accuracy harness helpers
// =============================================================================

// Print a labelled expected-vs-actual table with absolute and relative error,
// and return the worst relative error seen. Party 0 only.
double ReportAccuracy(int party_id, const std::string& label,
                      const std::vector<double>& inputs,
                      const std::vector<double>& expected,
                      const std::vector<double>& actual) {
    double worst_rel = 0.0, worst_abs = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double denom = std::abs(expected[i]) > 1e-12 ? std::abs(expected[i]) : 1.0;
        worst_abs = std::max(worst_abs, std::abs(expected[i] - actual[i]));
        worst_rel = std::max(worst_rel, std::abs(expected[i] - actual[i]) / denom);
    }
    if (party_id != 0) return worst_rel;

    std::cout << "\n--- " << label << " ---\n"
              << std::left << std::setw(14) << "Input" << std::setw(18) << "Plaintext"
              << std::setw(18) << "MPC" << std::setw(14) << "Abs Error"
              << std::setw(14) << "Rel Error" << std::endl;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double abs_err = std::abs(expected[i] - actual[i]);
        const double denom = std::abs(expected[i]) > 1e-12 ? std::abs(expected[i]) : 1.0;
        std::cout << std::left << std::setw(14) << inputs[i]
                  << std::setw(18) << expected[i] << std::setw(18) << actual[i]
                  << std::setw(14) << abs_err << std::setw(14) << (abs_err / denom)
                  << std::endl;
    }
    // Report both, because relative error is misleading near the bottom of the
    // fixed-point range: 1 ulp at precision 16 is 1.5e-5, so an expected value
    // of a few ulps (exp(-10) = 4.5e-5, say) can only ever be a few percent
    // accurate no matter how good the algorithm is. Absolute error at or below
    // ~1e-5 means the kernel is at the representation floor, which is the best
    // available answer.
    std::cout << "worst absolute error: " << std::scientific << worst_abs
              << "   worst relative error: " << worst_rel << std::defaultfloat << std::endl;
    return worst_rel;
}

// Exercises Div / Recip / Sqrt / Rsqrt against the standard library. These are
// the kernels this program adds, so they are checked before anything is built
// on top of them.
void TestNewKernels(EngineRef engine, int party_id) {
    single_cout("\n================ new fixed-point kernels ================");

    // Div spans four orders of magnitude in the denominator, which is what the
    // normalisation ladder exists to handle.
    const std::vector<double> num = {1.0, 3.0, -7.5, 1.0,   250.0, 1.0,    9.0,   -1.0};
    const std::vector<double> den = {1.0, 2.0,  4.0, 0.125, 1000.0, 0.03125, 7.0, 3.0};
    std::vector<double> div_expected(num.size()), div_labels(num.size());
    for (size_t i = 0; i < num.size(); ++i) {
        div_expected[i] = num[i] / den[i];
        div_labels[i] = den[i];
    }
    AV div_res = Div(ShareDoubles(engine, num), ShareDoubles(engine, den));
    ReportAccuracy(party_id, "Div (label = denominator)", div_labels, div_expected,
                   OpenToDoubles(div_res));

    const std::vector<double> rec_in = {0.03125, 0.1, 0.5, 1.0, 1.5, 2.0, 7.0, 100.0, 4096.0};
    std::vector<double> rec_expected(rec_in.size());
    for (size_t i = 0; i < rec_in.size(); ++i) rec_expected[i] = 1.0 / rec_in[i];
    ReportAccuracy(party_id, "Recip", rec_in, rec_expected,
                   OpenToDoubles(Recip(ShareDoubles(engine, rec_in))));

    const std::vector<double> sq_in = {0.01, 0.25, 0.5, 1.0, 2.0, 3.0, 10.0, 100.0, 1024.0, 65536.0};
    std::vector<double> sq_expected(sq_in.size()), rsq_expected(sq_in.size());
    for (size_t i = 0; i < sq_in.size(); ++i) {
        sq_expected[i] = std::sqrt(sq_in[i]);
        rsq_expected[i] = 1.0 / std::sqrt(sq_in[i]);
    }
    SqrtPair sp = SqrtBoth(ShareDoubles(engine, sq_in));
    ReportAccuracy(party_id, "Sqrt", sq_in, sq_expected, OpenToDoubles(sp.root));
    ReportAccuracy(party_id, "Rsqrt", sq_in, rsq_expected, OpenToDoubles(sp.inv_root));

    const std::vector<double> exp_in = {-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0};
    std::vector<double> exp_expected(exp_in.size());
    for (size_t i = 0; i < exp_in.size(); ++i) exp_expected[i] = std::exp(exp_in[i]);
    ReportAccuracy(party_id, "Exp", exp_in, exp_expected,
                   OpenToDoubles(Exp(ShareDoubles(engine, exp_in))));

    const std::vector<double> log_in = {0.01, 0.1, 0.25, 0.5, 0.7071, 1.0, 1.4142, 2.0, 4.0, 10.0, 100.0, 1000.0};
    std::vector<double> log_expected(log_in.size());
    for (size_t i = 0; i < log_in.size(); ++i) log_expected[i] = std::log(log_in[i]);
    ReportAccuracy(party_id, "Log", log_in, log_expected,
                   OpenToDoubles(Log(ShareDoubles(engine, log_in))));

    const std::vector<double> sig_in = {-8.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 8.0};
    std::vector<double> sig_expected(sig_in.size()), sp_expected(sig_in.size());
    for (size_t i = 0; i < sig_in.size(); ++i) {
        sig_expected[i] = sig_in[i] >= 0.0 ? 1.0 / (1.0 + std::exp(-sig_in[i]))
                                           : std::exp(sig_in[i]) / (1.0 + std::exp(sig_in[i]));
        sp_expected[i] = sig_in[i] > 0.0 ? sig_in[i] + std::log1p(std::exp(-sig_in[i]))
                                         : std::log1p(std::exp(sig_in[i]));
    }
    ReportAccuracy(party_id, "Sigmoid", sig_in, sig_expected,
                   OpenToDoubles(Sigmoid(ShareDoubles(engine, sig_in))));
    ReportAccuracy(party_id, "LogOnePlusExp (softplus)", sig_in, sp_expected,
                   OpenToDoubles(LogOnePlusExp(ShareDoubles(engine, sig_in))));

    const std::vector<double> abs_in = {-3.5, -1.0, -0.25, 0.0, 0.25, 1.0, 3.5};
    std::vector<double> abs_expected(abs_in.size());
    for (size_t i = 0; i < abs_in.size(); ++i) abs_expected[i] = std::abs(abs_in[i]);
    ReportAccuracy(party_id, "Abs", abs_in, abs_expected,
                   OpenToDoubles(Abs(ShareDoubles(engine, abs_in))));
}

// Checks the segmented-scan layer against a plaintext group-by on a RAGGED key
// column. This is the piece most likely to be subtly wrong -- the forward /
// reverse composition has a trap in it -- so it is verified directly rather
// than only through the models that use it.
void TestSegmented(EngineRef engine, int party_id) {
    single_cout("\n================ segmented (per-group) helpers ================");

    // Ragged groups: sizes 3, 2, 1, 4. Padded to a power of two with a sentinel
    // key so the pad block is its own segment on the reverse pass, and with
    // zeroed values so it contributes nothing.
    const std::vector<DataType> plain_keys_real = {1, 1, 1, 2, 2, 3, 4, 4, 4, 4};
    const std::vector<double> plain_vals_real = {1.5, 2.0, -0.5, 4.0, 1.0,
                                                 7.25, 0.5, 0.5, 0.5, 0.5};
    const std::vector<DataType> plain_mask_real = {1, 0, 0, 0, 0, 1, 1, 1, 0, 0};

    const size_t n_real = plain_keys_real.size();
    const size_t n = NextPowerOfTwo(n_real);
    const DataType kKeySentinel = std::numeric_limits<DataType>::max();

    cdough::Vector<DataType> kv(n, 0), vv(n, precision), mv(n, 0);
    for (size_t i = 0; i < n; ++i) {
        kv[i] = i < n_real ? plain_keys_real[i] : kKeySentinel;
        vv[i] = i < n_real ? static_cast<DataType>(std::llround(plain_vals_real[i] * scale)) : 0;
        mv[i] = i < n_real ? plain_mask_real[i] : 0;
    }

    BV key_col = engine.template secret_share_b<DataType>(kv, 0);
    AV val_col = engine.template secret_share_a<DataType>(vv, 0, precision);
    AV mask_col = engine.template secret_share_a<DataType>(mv, 0, 0);
    std::vector<BV> keys{key_col};

    // --- expected values, in the clear ---
    std::map<DataType, double> group_sum;
    std::map<DataType, int> group_count;
    std::map<DataType, double> group_first;
    std::set<DataType> distinct_masked;
    for (size_t i = 0; i < n_real; ++i) {
        group_sum[plain_keys_real[i]] += plain_vals_real[i];
        group_count[plain_keys_real[i]]++;
        if (group_count[plain_keys_real[i]] == 1)
            group_first[plain_keys_real[i]] = plain_vals_real[i];
        if (plain_mask_real[i]) distinct_masked.insert(plain_keys_real[i]);
    }

    AV total = SegTotal(keys, val_col);
    auto opened_total = total.open();
    AV prefix = SegScan(keys, val_col, SegDirection::Forward);
    auto opened_prefix = prefix.open();

    // The cached-group-bit scan reimplements the Brent-Kung level geometry, so
    // it is checked against the library's own aggregate rather than only
    // against the plaintext expectation: an index slip that happened to be
    // self-consistent would otherwise pass.
    ScanPlan plan = BuildScanPlan(key_col);
    std::vector<AV> plan_in{val_col};
    std::vector<AV> plan_total, plan_prefix;
    plan_total.emplace_back(n, engine);
    plan_prefix.emplace_back(n, engine);
    SegTotalPlanned(plan, plan_in, plan_total);
    SegScanPlanned(plan, plan_in, plan_prefix, SegDirection::Forward);
    auto opened_plan_total = plan_total[0].open();
    auto opened_plan_prefix = plan_prefix[0].open();
    BV last_b = LastOfGroup(keys);
    auto opened_last = last_b.open();
    BV first_b = FirstOfGroup(keys);
    auto opened_first = first_b.open();
    AV n_distinct = CountDistinct(keys, mask_col);
    auto opened_nd = n_distinct.open();

    // --- oblivious rank (ROW_NUMBER within key) ---
    // A column of ones on the real rows is the entire input: forward leaves the
    // 1-based position from the start of the group, reverse from the far end.
    cdough::Vector<DataType> ov(n, 0);
    for (size_t i = 0; i < n; ++i) ov[i] = i < n_real ? 1 : 0;
    AV ones_col = engine.template secret_share_a<DataType>(ov, 0, 0);
    ones_col.setPrecision(0);

    AV rank_fwd = SegRank(keys, ones_col, SegDirection::Forward);
    AV rank_rev = SegRank(keys, ones_col, SegDirection::Reverse);
    AV rank_plan = SegRankPlanned(plan, ones_col, SegDirection::Forward);

    AV first_bit = *(first_b.b2a_bit());
    first_bit.setPrecision(0);
    AV first_arith = *(first_bit * ones_col);
    first_arith.setPrecision(0);
    AV first_val = SegFirstValue(keys, val_col, first_arith);

    auto opened_rank_fwd = rank_fwd.open();
    auto opened_rank_rev = rank_rev.open();
    auto opened_rank_plan = rank_plan.open();
    auto opened_first_val = first_val.open();

    if (party_id != 0) return;

    std::cout << "\n--- SegTotal / SegScan / LastOfGroup (label = key) ---\n"
              << std::left << std::setw(6) << "row" << std::setw(6) << "key"
              << std::setw(10) << "value" << std::setw(14) << "total(exp)"
              << std::setw(14) << "total(mpc)" << std::setw(14) << "prefix(exp)"
              << std::setw(14) << "prefix(mpc)" << std::setw(7) << "first"
              << std::setw(6) << "last" << std::endl;

    bool ok = true;
    double running = 0.0;
    for (size_t i = 0; i < n_real; ++i) {
        if (i == 0 || plain_keys_real[i] != plain_keys_real[i - 1]) running = 0.0;
        running += plain_vals_real[i];
        const double exp_total = group_sum[plain_keys_real[i]];
        const double got_total = static_cast<double>(opened_total[i]) / scale;
        const double got_prefix = static_cast<double>(opened_prefix[i]) / scale;
        const bool exp_first = (i == 0 || plain_keys_real[i] != plain_keys_real[i - 1]);
        const bool exp_last =
            (i + 1 == n_real) || (plain_keys_real[i] != plain_keys_real[i + 1]);

        const double planned_total = static_cast<double>(opened_plan_total[i]) / scale;
        const double planned_prefix = static_cast<double>(opened_plan_prefix[i]) / scale;

        ok &= std::abs(exp_total - got_total) < 1e-3;
        ok &= std::abs(running - got_prefix) < 1e-3;
        // The cached scan must agree with the library's aggregate exactly.
        ok &= std::abs(planned_total - got_total) < 1e-9;
        ok &= std::abs(planned_prefix - got_prefix) < 1e-9;
        ok &= (static_cast<int>(opened_first[i]) == static_cast<int>(exp_first));
        // The last real row is only "last" once the sentinel block is excluded,
        // which is what makes the pad key matter.
        ok &= (static_cast<int>(opened_last[i]) == static_cast<int>(exp_last));

        std::cout << std::left << std::setw(6) << i << std::setw(6) << plain_keys_real[i]
                  << std::setw(10) << plain_vals_real[i] << std::setw(14) << exp_total
                  << std::setw(14) << got_total << std::setw(14) << running
                  << std::setw(14) << got_prefix << std::setw(7)
                  << static_cast<int>(opened_first[i]) << std::setw(6)
                  << static_cast<int>(opened_last[i]) << std::endl;
    }

    std::cout << "\n--- SegRank / SegFirstValue (ROW_NUMBER within key) ---\n"
              << std::left << std::setw(6) << "row" << std::setw(6) << "key"
              << std::setw(10) << "fwd(exp)" << std::setw(10) << "fwd(mpc)"
              << std::setw(10) << "rev(exp)" << std::setw(10) << "rev(mpc)"
              << std::setw(12) << "first(exp)" << std::setw(12) << "first(mpc)"
              << std::endl;

    long fwd = 0;
    for (size_t i = 0; i < n_real; ++i) {
        if (i == 0 || plain_keys_real[i] != plain_keys_real[i - 1]) fwd = 0;
        ++fwd;
        const long m = static_cast<long>(group_count[plain_keys_real[i]]);
        const long exp_rev = m - fwd + 1;  // rank_fwd + rank_rev == m + 1
        const long got_fwd = static_cast<long>(opened_rank_fwd[i]);
        const long got_rev = static_cast<long>(opened_rank_rev[i]);
        const long got_plan = static_cast<long>(opened_rank_plan[i]);
        const double exp_fv = group_first[plain_keys_real[i]];
        const double got_fv = static_cast<double>(opened_first_val[i]) / scale;

        ok &= (got_fwd == fwd);
        ok &= (got_rev == exp_rev);
        // The cached Brent-Kung network must agree with aggregate() exactly.
        ok &= (got_plan == got_fwd);
        ok &= std::abs(exp_fv - got_fv) < 1e-3;

        std::cout << std::left << std::setw(6) << i << std::setw(6) << plain_keys_real[i]
                  << std::setw(10) << fwd << std::setw(10) << got_fwd
                  << std::setw(10) << exp_rev << std::setw(10) << got_rev
                  << std::setw(12) << exp_fv << std::setw(12) << got_fv << std::endl;
    }

    const long got_nd = static_cast<long>(opened_nd[0]);
    const long exp_nd = static_cast<long>(distinct_masked.size());
    ok &= (got_nd == exp_nd);
    std::cout << "COUNT(DISTINCT key) WHERE mask=1: expected " << exp_nd << ", got " << got_nd
              << std::endl;
    std::cout << "cached-group-bit scan matches aggregate(): "
              << (std::abs(static_cast<double>(opened_plan_total[0]) -
                           static_cast<double>(opened_total[0])) < 1.0
                      ? "yes"
                      : "NO")
              << std::endl;
    std::cout << (ok ? "SEGMENTED HELPERS: PASS" : "SEGMENTED HELPERS: *** FAIL ***")
              << std::endl;
}

// =============================================================================
// Comparing two sequencings
//
// The three sources -- MPC, SequencePlain, SQLite -- must agree, but NOT
// positionally. Bitonic sort is not stable, and SQLite's ROW_NUMBER over rows
// tied on (subject_id, encounter_dt) is likewise arbitrary, so "row i of one
// equals row i of the other" is not a well-formed claim on a patient with two
// encounters on one date.
//
// What IS determined is the value at each RANK. Fix a patient and a scope, let
// the sorted date sequence be d(1) <= ... <= d(n). Whichever tied row lands at
// position k gets visit_num = k, index_visit = [k == 1], final_visit = [k == n],
// index_dt = d(1), and fu_month = band(d(k) - d(1)). Every one of those is a
// function of k and of the sorted sequence alone, and the sorted sequence of a
// multiset is unique -- so the whole tuple at rank k is identical across all
// legal tie orders.
//
// Hence: group by subject_id, order each group by visit_num, compare
// element-wise, and additionally assert visit_num is exactly 1..n. That last
// check is why this is stronger than comparing multisets -- a multiset compare
// cannot see a duplicated or skipped rank, which is exactly what a wrong
// Brent-Kung group bit would produce.
//
// This replaces an earlier positional alignment that tiebroke on data_source.
// That only helped when the tied rows were in different systems, and gave false
// confidence otherwise.
// =============================================================================

struct SeqDiff {
    long lhs_rows = 0, rhs_rows = 0;
    long missing_subjects = 0;  // in lhs, absent from rhs
    long extra_subjects = 0;    // in rhs, absent from lhs
    long ragged_runs = 0;       // same patient, different number of encounters
    long bad_rank = 0;          // visit_num was not 1..n
    long bad_dt = 0, bad_index = 0, bad_final = 0, bad_fu = 0;
    long bad_payload = 0;
    long tie_groups = 0;  // (subject, date) pairs carrying more than one row

    bool ok() const {
        return lhs_rows == rhs_rows && missing_subjects == 0 && extra_subjects == 0 &&
               ragged_runs == 0 && bad_rank == 0 && bad_dt == 0 && bad_index == 0 &&
               bad_final == 0 && bad_fu == 0 && bad_payload == 0;
    }
};

// Group row indices by subject, each group ordered by visit_num.
inline std::map<DataType, std::vector<size_t>> GroupBySubject(const PlainCohort& c) {
    std::map<DataType, std::vector<size_t>> by_subject;
    for (size_t i = 0; i < c.rows(); ++i) by_subject[c.subject_id[i]].push_back(i);
    for (auto& [sid, rows] : by_subject)
        std::sort(rows.begin(), rows.end(),
                  [&](size_t x, size_t y) { return c.visit_num[x] < c.visit_num[y]; });
    return by_subject;
}

// `strict_payload` additionally compares newage / data_source / sisa / sa. Off by
// default, and it should stay off on tied data: the pairing of a RANK to a
// PAYLOAD is genuinely ambiguous there. Two same-day rows with different sisa
// get ranks k and k+1 in whichever order the sort happened to pick, and no
// implementation is more right than the other. (gender and hispanic are one-hot
// in SecureCohort and are not reconstructed, so they are never compared.)
SeqDiff CompareSequencing(const PlainCohort& lhs, const PlainCohort& rhs,
                          bool strict_payload = false) {
    SeqDiff d;
    d.lhs_rows = static_cast<long>(lhs.rows());
    d.rhs_rows = static_cast<long>(rhs.rows());

    std::map<std::pair<DataType, DataType>, long> seen;
    for (size_t i = 0; i < lhs.rows(); ++i) ++seen[{lhs.subject_id[i], lhs.encounter_dt[i]}];
    for (const auto& [key, n] : seen)
        if (n > 1) ++d.tie_groups;

    const auto l_by = GroupBySubject(lhs);
    const auto r_by = GroupBySubject(rhs);

    for (const auto& [sid, l_rows] : l_by) {
        const auto it = r_by.find(sid);
        if (it == r_by.end()) {
            ++d.missing_subjects;
            continue;
        }
        const std::vector<size_t>& r_rows = it->second;
        if (l_rows.size() != r_rows.size()) {
            ++d.ragged_runs;
            continue;
        }
        const long n = static_cast<long>(l_rows.size());
        for (long k = 0; k < n; ++k) {
            const size_t li = l_rows[k], ri = r_rows[k];

            // The ranks must be a permutation-free 1..n on BOTH sides.
            if (lhs.visit_num[li] != k + 1 || rhs.visit_num[ri] != k + 1) ++d.bad_rank;

            if (lhs.encounter_dt[li] != rhs.encounter_dt[ri]) ++d.bad_dt;
            if (lhs.index_visit[li] != rhs.index_visit[ri]) ++d.bad_index;
            if (lhs.final_visit[li] != rhs.final_visit[ri]) ++d.bad_final;

            // Compare fu_month as `present ? value : NULL`, so the value is
            // ignored wherever the pipeline calls it absent.
            const DataType l_fu = lhs.fu_month_present[li] ? lhs.fu_month[li] : -1;
            const DataType r_fu = rhs.fu_month_present[ri] ? rhs.fu_month[ri] : -1;
            if (l_fu != r_fu) ++d.bad_fu;

            if (strict_payload) {
                if (lhs.newage[li] != rhs.newage[ri] ||
                    lhs.data_source[li] != rhs.data_source[ri] ||
                    lhs.sisa[li] != rhs.sisa[ri] || lhs.sa[li] != rhs.sa[ri])
                    ++d.bad_payload;
            }
        }
    }
    for (const auto& [sid, r_rows] : r_by)
        if (l_by.find(sid) == l_by.end()) ++d.extra_subjects;

    return d;
}

// Party 0 only. Prints one line when clean, the offending counters when not.
void ReportSeqDiff(const std::string& label, const SeqDiff& d) {
    if (d.ok()) {
        std::cout << "  " << std::left << std::setw(34) << label << "MATCH  (" << d.lhs_rows
                  << " rows)" << std::endl;
        return;
    }
    std::cout << "  " << std::left << std::setw(34) << label << "*** MISMATCH ***"
              << "\n    rows " << d.lhs_rows << " vs " << d.rhs_rows;
    auto field = [](const char* name, long v) {
        if (v) std::cout << "\n    " << name << " " << v;
    };
    field("subjects missing from rhs", d.missing_subjects);
    field("subjects only in rhs     ", d.extra_subjects);
    field("ragged patient runs      ", d.ragged_runs);
    field("visit_num not 1..n       ", d.bad_rank);
    field("encounter_dt mismatches  ", d.bad_dt);
    field("index_visit mismatches   ", d.bad_index);
    field("final_visit mismatches   ", d.bad_final);
    field("fu_month mismatches      ", d.bad_fu);
    field("payload mismatches       ", d.bad_payload);
    std::cout << std::endl;
}

// Open a secret-shared cohort into the plaintext layout the comparator takes.
//
// COLLECTIVE: open() is a protocol operation, so EVERY party must call this or
// the run deadlocks. Only party 0 gets a populated result.
//
// Descaling lives here rather than at each call site. Per primitives.h, visit_num,
// fu_month and newage are scaled by `scale`; encounter_dt, valid and the 0/1
// flags are not.
PlainCohort CohortFromSecure(const SecureCohort& sc, int party_id) {
    auto o_subject = sc.subject_key.open();
    auto o_dt = sc.encounter_dt.open();
    auto o_valid = sc.valid.open();
    auto o_visit = sc.visit_num.open();
    auto o_index = sc.index_visit.open();
    auto o_final = sc.final_visit.open();
    auto o_fu = sc.fu_month.open();
    auto o_present = sc.fu_present.open();
    auto o_umass = sc.umass.open();
    auto o_newage = sc.newage.open();
    auto o_sisa = sc.sisa.open();
    auto o_sa = sc.sa.open();

    PlainCohort c;
    c.scope = sc.scope;
    if (party_id != 0) return c;

    const auto S = static_cast<DataType>(scale);
    for (size_t i = 0; i < o_valid.size(); ++i) {
        if (static_cast<DataType>(o_valid[i]) != 1) continue;  // pad row
        c.subject_id.push_back(static_cast<DataType>(o_subject[i]));
        c.encounter_dt.push_back(static_cast<DataType>(o_dt[i]));
        c.newage.push_back(static_cast<DataType>(o_newage[i]) / S);
        c.gender.push_back(0);    // one-hot in SecureCohort; not reconstructed
        c.hispanic.push_back(0);  // likewise
        c.index_visit.push_back(static_cast<DataType>(o_index[i]));
        c.visit_num.push_back(static_cast<DataType>(o_visit[i]) / S);
        c.final_visit.push_back(static_cast<DataType>(o_final[i]));
        c.fu_month.push_back(static_cast<DataType>(o_fu[i]) / S);
        c.fu_month_present.push_back(static_cast<DataType>(o_present[i]));
        c.data_source.push_back(static_cast<DataType>(o_umass[i]) == 1 ? 1 : 2);
        c.sisa.push_back(static_cast<DataType>(o_sisa[i]));
        c.sa.push_back(static_cast<DataType>(o_sa[i]));
    }
    return c;
}

// Checks the whole cross-party half -- oblivious merge, then both sequencing
// passes -- against TWO independent oracles.
//
// This is the test that matters for tasks/0011, and since tasks/0012 it is also
// where the SQL cross-check lives. SequencePlain and the MPC implementation were
// written from the same reading of the same upstream query, so agreement between
// them is weaker evidence than it looks; SQLite executes that query AS SQL and
// is independent of both. Each scope is therefore checked three ways:
//
//     SQL  vs SequencePlain    -- audits the hand-written oracle
//     SQL  vs MPC              -- audits the secure implementation
//     MPC  vs SequencePlain    -- the original check, retained
//
// Always on the synthetic path: it generates and splits its own base table, so
// the union both oracles need genuinely exists here.
void TestTwoOwnerPipeline(EngineRef engine, int party_id, size_t subjects, int party_a,
                          int party_b) {
    single_cout("\n================ two-owner merge and sequencing ================");

    GenTruth truth;
    PlainBaseTable base = GenerateBaseTable(subjects, truth, 424242);
    OwnerSplit split = SplitAcrossOwners(base, 0.6, 0.0, 99);
    ApplyMrnRepair(split.a);
    ApplyMrnRepair(split.b);
    PlainFlagged fa = DeriveFlagged(split.a);
    PlainFlagged fb = DeriveFlagged(split.b);

    SecurePipeline sp =
        RunSecurePipeline(engine, fa, party_a, fb, party_b, fa.rows(), fb.rows());

    // Collective: every party must reach all three, in this order.
    const PlainCohort mpc_any = CohortFromSecure(sp.any, party_id);
    const PlainCohort mpc_umass = CohortFromSecure(sp.umass, party_id);
    const PlainCohort mpc_nonumass = CohortFromSecure(sp.nonumass, party_id);

    if (party_id != 0) return;

    bool ok = true;
    const std::pair<SystemScope, const PlainCohort*> scopes[] = {
        {SystemScope::Any, &mpc_any},
        {SystemScope::UMass, &mpc_umass},
        {SystemScope::NonUMass, &mpc_nonumass},
    };

#if defined(HAVE_SQLITE3)
    SqlOracle sql(fa, fb);
    if (!sql.ok()) {
        std::cout << "  *** the SQL oracle failed to build; SQL cross-check skipped ***"
                  << std::endl;
        ok = false;
    }
    const long same_date_ties = sql.ok() ? sql.CountSameDateTies() : -1;
#else
    std::cout << "  NOTE: built without SQLite (HAVE_SQLITE3 undefined), so the SQL\n"
                 "        cross-check is ABSENT. Only MPC vs SequencePlain runs below."
              << std::endl;
#endif

    for (const auto& [scope, mpc] : scopes) {
        std::cout << "\n  --- " << ScopeName(scope) << " ---" << std::endl;
        const PlainCohort plain = SequencePlain(fa, fb, scope);

#if defined(HAVE_SQLITE3)
        if (sql.ok()) {
            const PlainCohort q = sql.Sequence(scope);
            const SeqDiff d_plain = CompareSequencing(q, plain);
            const SeqDiff d_mpc = CompareSequencing(q, *mpc);
            ReportSeqDiff("SQL vs SequencePlain", d_plain);
            ReportSeqDiff("SQL vs MPC", d_mpc);
            ok &= d_plain.ok() && d_mpc.ok();
        }
#endif
        const SeqDiff d_both = CompareSequencing(plain, *mpc);
        ReportSeqDiff("SequencePlain vs MPC", d_both);
        ok &= d_both.ok();
    }

#if defined(HAVE_SQLITE3)
    // A PASS on a run that contained no ties is no evidence that tie handling
    // works, so say how much tie coverage this run actually had.
    std::cout << "\n  tie coverage: " << same_date_ties
              << " (patient, date) pairs carrying more than one encounter" << std::endl;
#endif

    std::cout << (ok ? "TWO-OWNER PIPELINE: PASS" : "TWO-OWNER PIPELINE: *** FAIL ***")
              << std::endl;
}

// Checks Gram / CholeskySolve / SymmetricInverse against plaintext linear
// algebra: the solve is scored by the residual ||Ax - b||_inf and the inverse by
// ||A A^-1 - I||_inf, both computed in the clear from the opened results.
void TestLinearAlgebra(EngineRef engine, int party_id) {
    single_cout("\n================ dense secure linear algebra ================");

    const size_t n = 64;
    const size_t p = 4;

    // A small design matrix and weights, chosen so X^T W X is comfortably SPD.
    std::vector<double> x_plain(n * p), w_plain(n), y_plain(n);
    for (size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        x_plain[i * p + 0] = 1.0;
        x_plain[i * p + 1] = t - 0.5;
        x_plain[i * p + 2] = (i % 3 == 0) ? 1.0 : 0.0;
        x_plain[i * p + 3] = std::sin(6.0 * t) * 0.5;
        w_plain[i] = 0.1 + 0.15 * ((i % 5) / 4.0);
        y_plain[i] = 0.3 + 0.8 * x_plain[i * p + 1] - 0.4 * x_plain[i * p + 2];
    }

    std::vector<double> gram_expected(p * p, 0.0);
    for (size_t k = 0; k < p; ++k)
        for (size_t l = 0; l < p; ++l)
            for (size_t i = 0; i < n; ++i)
                gram_expected[k * p + l] += w_plain[i] * x_plain[i * p + k] * x_plain[i * p + l];
    // A mild ridge, matching what the model code will use.
    const double lambda = 1e-3;
    for (size_t k = 0; k < p; ++k) gram_expected[k * p + k] += lambda;

    std::vector<double> b_plain(p, 0.0);
    for (size_t k = 0; k < p; ++k)
        for (size_t i = 0; i < n; ++i) b_plain[k] += x_plain[i * p + k] * y_plain[i];

    AV x_sec = ShareDoubles(engine, x_plain);
    AV w_sec = ShareDoubles(engine, w_plain);
    AV b_sec = ShareDoubles(engine, b_plain);

    AV g = Gram(x_sec, w_sec, n, p);
    AddRidge(g, p, lambda);
    AV sol = CholeskySolve(g, b_sec, p);
    AV inv = SymmetricInverse(g, p);

    const std::vector<double> g_open = OpenToDoubles(g);
    const std::vector<double> sol_open = OpenToDoubles(sol);
    const std::vector<double> inv_open = OpenToDoubles(inv);

    if (party_id != 0) return;

    double gram_err = 0.0;
    for (size_t i = 0; i < p * p; ++i)
        gram_err = std::max(gram_err, std::abs(g_open[i] - gram_expected[i]));

    // Residual of the secure solution, measured against the EXACT Gram matrix.
    double resid = 0.0;
    for (size_t k = 0; k < p; ++k) {
        double row = 0.0;
        for (size_t l = 0; l < p; ++l) row += gram_expected[k * p + l] * sol_open[l];
        resid = std::max(resid, std::abs(row - b_plain[k]));
    }

    double inv_err = 0.0;
    for (size_t k = 0; k < p; ++k) {
        for (size_t l = 0; l < p; ++l) {
            double acc = 0.0;
            for (size_t m = 0; m < p; ++m) acc += gram_expected[k * p + m] * inv_open[m * p + l];
            inv_err = std::max(inv_err, std::abs(acc - (k == l ? 1.0 : 0.0)));
        }
    }

    std::cout << "\n--- Gram / Cholesky (n=" << n << ", p=" << p << ") ---\n";
    std::cout << "max |Gram_mpc - Gram_plain|      : " << std::scientific << gram_err << "\n";
    std::cout << "solve residual ||Ax - b||_inf    : " << resid << "\n";
    std::cout << "inverse error ||A A^-1 - I||_inf : " << inv_err << std::defaultfloat
              << std::endl;

    std::cout << "\nsolution (mpc vs plaintext):\n";
    // Reference solution by plaintext Gaussian elimination on the exact Gram.
    std::vector<double> m(gram_expected), rhs(b_plain);
    for (size_t c = 0; c < p; ++c) {
        size_t piv = c;
        for (size_t r = c + 1; r < p; ++r)
            if (std::abs(m[r * p + c]) > std::abs(m[piv * p + c])) piv = r;
        for (size_t k = 0; k < p; ++k) std::swap(m[c * p + k], m[piv * p + k]);
        std::swap(rhs[c], rhs[piv]);
        const double d = m[c * p + c];
        for (size_t k = 0; k < p; ++k) m[c * p + k] /= d;
        rhs[c] /= d;
        for (size_t r = 0; r < p; ++r) {
            if (r == c) continue;
            const double f = m[r * p + c];
            for (size_t k = 0; k < p; ++k) m[r * p + k] -= f * m[c * p + k];
            rhs[r] -= f * rhs[c];
        }
    }
    bool ok = (gram_err < 1e-2) && (resid < 1e-2) && (inv_err < 1e-2);
    for (size_t k = 0; k < p; ++k) {
        std::cout << "  x[" << k << "] = " << std::setw(12) << sol_open[k]
                  << "   plaintext " << std::setw(12) << rhs[k]
                  << "   diff " << std::abs(sol_open[k] - rhs[k]) << std::endl;
        ok &= std::abs(sol_open[k] - rhs[k]) < 1e-2;
    }
    std::cout << (ok ? "LINEAR ALGEBRA: PASS" : "LINEAR ALGEBRA: *** FAIL ***") << std::endl;
}

// Measures the cost of the pieces a mixed-model fit is built from, so an
// optimisation can be checked against a number instead of an intuition.
void BenchmarkObjective(EngineRef engine, int party_id, const SecureCohort& cohort) {
    using Clock = std::chrono::steady_clock;
    auto seconds_since = [](Clock::time_point start) {
        return std::chrono::duration<double>(Clock::now() - start).count();
    };

    // Model 5a is the widest mixed model: six fixed effects plus the variance.
    ModelSpec spec;
    spec.step = "5a";
    spec.time = TimeAxis::VisitNum;
    spec.covars = true;
    spec.scope = cohort.scope;
    ModelData md = BuildDesign(cohort, spec);
    const size_t dim = md.p + 1;

    std::vector<AV> params;
    for (size_t k = 0; k < dim; ++k) {
        AV v(1, engine);
        v.setPrecision(precision);
        params.push_back(v);
    }
    auto objective = [&md](const std::vector<AV>& p) { return FlatNegMarginalLogLik(md, p); };

    // One untimed call first, so any lazily built correlated randomness is not
    // charged to the measurement.
    objective(params).open();

    g_objective_evaluations = 0;
    auto start = Clock::now();
    objective(params).open();
    const double one_eval = seconds_since(start);

    start = Clock::now();
    std::vector<AV> numeric = NumericalGradient(objective, params);
    std::vector<double> numeric_open;
    for (AV& g : numeric) numeric_open.push_back(OpenScalar(g));
    const double numeric_gradient = seconds_since(start);
    const long evals_in_gradient = g_objective_evaluations - 1;

    // The analytic gradient has to be checked against the numerical one before
    // it can be trusted: a sign slip or a missing term in the derivation would
    // otherwise show up only as a mysteriously worse fit.
    g_objective_evaluations = 0;
    start = Clock::now();
    std::vector<AV> analytic;
    AV value = FlatObjective(md, params, &analytic);
    value.open();
    std::vector<double> analytic_open;
    for (AV& g : analytic) analytic_open.push_back(OpenScalar(g));
    const double analytic_gradient = seconds_since(start);
    const long evals_in_analytic = g_objective_evaluations;

    start = Clock::now();
    std::vector<AV> in{md.y, md.row_mask};
    std::vector<AV> out;
    out.emplace_back(md.n_pad, engine);
    out.emplace_back(md.n_pad, engine);
    SegTotal(md.keys, in, out);
    out[0].open();
    const double one_segtotal = seconds_since(start);

    // Library aggregate versus the cached-group-bit scan, at both the 2-column
    // width the Newton loop uses and the 3p+2 width the gradient uses.
    auto time_scan = [&](size_t width, bool planned, bool total) {
        std::vector<AV> cols, dst;
        for (size_t k = 0; k < width; ++k) {
            cols.push_back(md.y);
            dst.emplace_back(md.n_pad, engine);
        }
        std::vector<BV> ks = md.keys;
        auto t0 = Clock::now();
        if (total) {
            if (planned) SegTotalPlanned(*md.scan_plan, cols, dst);
            else SegTotal(ks, cols, dst);
        } else {
            if (planned) SegScanPlanned(*md.scan_plan, cols, dst, SegDirection::Forward);
            else SegScan(ks, cols, dst, SegDirection::Forward);
        }
        dst[0].open();
        return seconds_since(t0);
    };
    const size_t wide = 3 * md.p + 2;
    const double lib_total2 = time_scan(2, false, true);
    const double plan_total2 = time_scan(2, true, true);
    const double lib_scan_wide = time_scan(wide, false, false);
    const double plan_scan_wide = time_scan(wide, true, false);

    start = Clock::now();
    ScanPlan fresh = BuildScanPlan(md.keys[0]);
    (void)fresh;
    const double plan_build = seconds_since(start);

    // Log is timed to bound what a --public-clusters fast path could win. The
    // per-cluster quantities -- the Newton-step reciprocal, 1/A and log(A) --
    // are computed on all n rows but only used on one row per cluster, so
    // revealing the cluster boundaries would let them run on m rows instead.
    AV logarg(md.n_pad, engine);
    logarg.setPrecision(0);
    logarg += DataType(scale) * 2;
    logarg.setPrecision(precision);
    start = Clock::now();
    AV lg = Log(logarg);
    lg.open();
    const double one_log = seconds_since(start);

    start = Clock::now();
    AV sig = Sigmoid(md.y);
    sig.open();
    const double one_sigmoid = seconds_since(start);

    // Split Sigmoid into its two halves. Sigmoid's denominator is 1 + exp(-|eta|),
    // which lies in [1, 2] exactly, so if Div's normalisation ladder is a large
    // share of the cost then a range-specialised reciprocal is worth having.
    start = Clock::now();
    AV ex = Exp(md.y);
    ex.open();
    const double one_exp = seconds_since(start);

    AV den(md.n_pad, engine);
    den.setPrecision(0);
    den += DataType(scale) + DataType(scale) / 2;  // a denominator inside [1, 2]
    den.setPrecision(precision);
    start = Clock::now();
    AV rec = Div(md.y, den);
    rec.open();
    const double one_div = seconds_since(start);

    if (party_id != 0) return;
    std::cout << "\n================ cost breakdown ================\n"
              << "population           " << ScopeName(cohort.scope) << "\n"
              << "rows (padded)        " << md.n_pad << "\n"
              << "parameters (dim)     " << dim << "\n"
              << std::fixed << std::setprecision(4)
              << "\n  one Exp over n rows          " << one_exp << " s"
              << "\n  one Div over n rows          " << one_div << " s"
              << "\n  one Sigmoid over n rows      " << one_sigmoid << " s  (Exp + Div + masks)"
              << "\n  one SegTotal (2 columns)     " << one_segtotal << " s"
              << "\n  SegTotal  2 cols: aggregate  " << lib_total2 << " s   cached "
              << plan_total2 << " s"
              << "\n  SegScan  " << wide << " cols: aggregate  " << lib_scan_wide
              << " s   cached " << plan_scan_wide << " s"
              << "\n  one Log over n rows          " << one_log << " s"
              << "\n  building the scan plan       " << plan_build << " s (once per cohort)"
              << "\n  one objective evaluation     " << one_eval << " s"
              << "\n  numerical gradient           " << numeric_gradient << " s  ("
              << evals_in_gradient << " objective evaluations)"
              << "\n  analytic gradient + value    " << analytic_gradient << " s  ("
              << evals_in_analytic << " objective evaluation)"
              << "\n  speedup                      "
              << (analytic_gradient > 0 ? numeric_gradient / analytic_gradient : 0.0) << "x"
              << std::defaultfloat << std::endl;

    std::cout << "\n  gradient agreement (analytic vs central differences)\n  " << std::left
              << std::setw(34) << "term" << std::setw(16) << "analytic" << std::setw(16)
              << "numerical" << std::setw(12) << "abs diff" << std::endl;
    double worst = 0.0, scale_of_grad = 0.0;
    for (size_t k = 0; k < analytic_open.size(); ++k) {
        const double d = std::abs(analytic_open[k] - numeric_open[k]);
        worst = std::max(worst, d);
        scale_of_grad = std::max(scale_of_grad, std::abs(numeric_open[k]));
        const std::string name = k < md.terms.size() ? md.terms[k] : "s (log sigma)";
        std::cout << "  " << std::left << std::setw(34) << name << std::fixed
                  << std::setprecision(6) << std::setw(16) << analytic_open[k] << std::setw(16)
                  << numeric_open[k] << std::setw(12) << d << std::defaultfloat << std::endl;
    }
    const double relative = scale_of_grad > 0 ? worst / scale_of_grad : worst;
    std::cout << "  worst |difference| = " << std::scientific << worst << "  (relative to the "
              << "largest component: " << relative << ")" << std::defaultfloat << "\n  "
              << (relative < 0.05 ? "ANALYTIC GRADIENT: PASS"
                                  : "ANALYTIC GRADIENT: *** CHECK ***")
              << "\n\n  a " << kGlmmBfgsIterations
              << "-iteration fit costs roughly " << std::fixed << std::setprecision(1)
              << (kGlmmBfgsIterations * (numeric_gradient + 2 * one_eval) + numeric_gradient)
              << " s with numerical gradients, "
              << (kGlmmBfgsIterations * (analytic_gradient + 2 * one_eval) + analytic_gradient)
              << " s with analytic ones" << std::defaultfloat << std::endl;

    // What is left on the table, and at what price.
    const double per_cluster_work =
        static_cast<double>(kNewtonIterations) * one_div + one_div + one_log;
    const double mean_cluster =
        static_cast<double>(md.rows_used) / std::max<double>(1.0, static_cast<double>(cohort.num_subjects));
    const double reclaimable = per_cluster_work * (1.0 - 1.0 / mean_cluster);
    std::cout << "\n  per-cluster work done on all n rows: " << std::fixed
              << std::setprecision(4) << per_cluster_work << " s of " << one_eval
              << " s (" << std::setprecision(0) << 100.0 * per_cluster_work / one_eval
              << "%)\n  mean cluster size " << std::setprecision(2) << mean_cluster
              << ", so revealing the cluster boundaries could reclaim at most "
              << std::setprecision(4) << reclaimable << " s -- a "
              << std::setprecision(2) << one_eval / (one_eval - reclaimable)
              << "x objective speedup." << std::defaultfloat << std::endl;
}

}  // namespace cdough::regression

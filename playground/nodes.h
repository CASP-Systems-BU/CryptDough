#pragma once

#include <string>
#include <vector>

#include "./cohort.h"
#include "./primitives.h"

// The descriptive and aggregate lineage nodes: d1a, d1b and the three
// sisa_perct_cnt tables, plus the summary statistics they are built from.

namespace cdough::regression {

// =============================================================================
// Descriptive nodes (d1a, d1b) and the aggregate tables (sisa_perct_cnt)
// =============================================================================


// Sum of a 0/1 mask: a count.
AV MaskCount(const AV& mask) {
    AV c = mask.chunkedSum(mask.size());
    c.setPrecision(0);
    return c;
}

// Sum of a scaled column over the masked rows.
AV MaskedSum(const AV& x_scaled, const AV& mask) {
    AV prod = *(x_scaled * mask);  // mask is an unscaled 0/1, so no rescale
    AV s = prod.chunkedSum(prod.size());
    s.setPrecision(0);
    return s;
}

// COUNT(*) of masked rows whose integer value is <= threshold.
AV MaskedCountAtMost(const AV& x_scaled, const AV& mask, long threshold) {
    AV diff = -x_scaled;
    diff += static_cast<DataType>(threshold) * DataType(scale);
    AV le = *(diff.gtez());  // 1 where x <= threshold
    AV hit = *(le * mask);
    AV c = hit.chunkedSum(hit.size());
    c.setPrecision(0);
    return c;
}

// The k-th smallest masked value (0-based), by binary search on the CDF over a
// PUBLIC integer range.
//
// One comparison bit is opened per step. The search path is a function of the
// returned order statistic and nothing else, so this discloses exactly the
// quantile that the pipeline publishes anyway -- not the underlying values, and
// not the rest of the distribution. That is the whole reason for preferring this
// to an oblivious sort: it is cheaper AND it leaks strictly less than revealing
// a full histogram would.
long SecureOrderStat(const AV& x_scaled, const AV& mask, long k, long lo, long hi) {
    while (lo < hi) {
        const long mid = lo + (hi - lo) / 2;
        const double cnt = OpenScalar(MaskedCountAtMost(x_scaled, mask, mid), false);
        if (cnt >= static_cast<double>(k) + 0.5) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

// PROC UNIVARIATE's moments and quantiles for one column over the masked rows.
struct Univariate {
    long n = 0;
    double mean = 0.0, sd = 0.0;
    double minimum = 0.0, q1 = 0.0, median = 0.0, q3 = 0.0, maximum = 0.0;
};

// Quantiles use PCTLDEF=4 (linear interpolation), which is what PERCENTILE_CONT
// computes. PROC UNIVARIATE defaults to PCTLDEF=5 (empirical with averaging), so
// on small samples the two can differ by one observation's worth; adding
// PCTLDEF=4 to the SAS call makes them agree.
Univariate SecureUnivariate(const AV& x_scaled, const AV& mask, long lo, long hi) {
    Univariate u;
    u.n = static_cast<long>(std::llround(OpenScalar(MaskCount(mask), false)));
    if (u.n <= 0) return u;

    AV n_sec = MaskCount(mask);
    n_sec.setPrecision(0);
    AV n_scaled = *(n_sec * DataType(scale));

    AV total = MaskedSum(x_scaled, mask);
    AV mean = Div(total, n_scaled);
    mean.setPrecision(0);
    u.mean = OpenScalar(mean);

    // Two-pass variance: subtract the mean first rather than forming sum(x^2).
    // Numerically safer, and it keeps the accumulator inside int64 for the
    // largest cohorts this program is meant to run on.
    AV centred = x_scaled - mean.repeated_subset_reference(x_scaled.size());
    centred.setPrecision(0);
    AV sq = *(*(centred * centred) / scale);
    AV ss = MaskedSum(sq, mask);
    if (u.n > 1) {
        AV denom(1, x_scaled.engine);
        denom.setPrecision(0);
        denom += static_cast<DataType>(u.n - 1) * DataType(scale);
        u.sd = std::sqrt(std::max(0.0, OpenScalar(Div(ss, denom))));
    }

    auto order_stat = [&](long k) {
        return static_cast<double>(SecureOrderStat(x_scaled, mask, k, lo, hi));
    };
    auto pctl = [&](double q) {
        const double h = (static_cast<double>(u.n) - 1.0) * q;
        const long lo_idx = static_cast<long>(std::floor(h));
        const double frac = h - static_cast<double>(lo_idx);
        const double a = order_stat(lo_idx);
        if (frac <= 0.0 || lo_idx + 1 >= u.n) return a;
        return a + frac * (order_stat(lo_idx + 1) - a);
    };

    u.minimum = order_stat(0);
    u.q1 = pctl(0.25);
    u.median = pctl(0.50);
    u.q3 = pctl(0.75);
    u.maximum = order_stat(u.n - 1);
    return u;
}

void PrintUnivariate(const std::string& label, const Univariate& u) {
    std::cout << "\n  " << label << "\n"
              << "    n       " << u.n << "\n"
              << "    mean    " << std::fixed << std::setprecision(4) << u.mean << "\n"
              << "    sd      " << u.sd << "\n"
              << "    min     " << u.minimum << "\n"
              << "    q1      " << u.q1 << "\n"
              << "    median  " << u.median << "\n"
              << "    q3      " << u.q3 << "\n"
              << "    max     " << u.maximum << std::defaultfloat << std::endl;
}

// A one-way frequency table over a PUBLIC level set. Cheaper than an oblivious
// group-by and it discloses nothing beyond the counts, which are the published
// output. Missing values are excluded from the denominator, matching PROC FREQ.
struct FreqRow {
    long value = 0;
    long frequency = 0;
    double percent = 0.0;
};

std::vector<FreqRow> SecureFrequency(const std::vector<AV>& indicators,
                                     const std::vector<DataType>& levels, const AV& mask) {
    std::vector<FreqRow> rows;
    long total = 0;
    for (size_t i = 0; i < levels.size(); ++i) {
        AV hit = *(indicators[i] * mask);
        const long f = static_cast<long>(std::llround(OpenScalar(MaskCount(hit), false)));
        rows.push_back(FreqRow{static_cast<long>(levels[i]), f, 0.0});
        total += f;
    }
    for (FreqRow& r : rows)
        r.percent = total > 0 ? 100.0 * static_cast<double>(r.frequency) / total : 0.0;
    return rows;
}

// The visit-count distribution: a frequency table over a public range of integer
// values. The counts ARE the published output of node d1a, so revealing them is
// the point.
std::vector<FreqRow> SecureValueHistogram(const AV& x_scaled, const AV& mask, long lo, long hi) {
    std::vector<FreqRow> rows;
    long total = 0;
    double prev_cdf = 0.0;
    for (long v = lo; v <= hi; ++v) {
        const double cdf = OpenScalar(MaskedCountAtMost(x_scaled, mask, v), false);
        const long f = static_cast<long>(std::llround(cdf - prev_cdf));
        prev_cdf = cdf;
        if (f > 0) rows.push_back(FreqRow{v, f, 0.0});
        total += f;
    }
    for (FreqRow& r : rows)
        r.percent = total > 0 ? 100.0 * static_cast<double>(r.frequency) / total : 0.0;
    return rows;
}

void PrintFrequency(const std::string& label, const std::string& value_header,
                    const std::vector<FreqRow>& rows) {
    std::cout << "\n  " << label << "\n    " << std::left << std::setw(22) << value_header
              << std::setw(12) << "frequency" << std::setw(10) << "percent" << std::endl;
    for (const FreqRow& r : rows)
        std::cout << "    " << std::left << std::setw(22) << r.value << std::setw(12)
                  << r.frequency << std::fixed << std::setprecision(2) << r.percent
                  << std::defaultfloat << std::endl;
}

// --- node d1a: visits per patient --------------------------------------------
//
// final_visit = 1 keeps exactly one row per patient -- their last encounter --
// and visit_num on that row is a running count, so it equals the patient total.
//
// `max_visits` is the PUBLIC upper bound of the histogram sweep and of the
// quantile binary search. It must be a run parameter, not a property of the
// data: it sets how many collective operations this node performs, so if the
// parties disagree about it they perform different numbers of rounds and the
// protocol desynchronises. Deriving it from a local plaintext maximum -- which
// only the owning party has -- produced exactly that failure. Sweeping past the
// largest actual value is harmless: empty buckets are dropped from the output.
void ReportD1a(const SecureCohort& c, int party_id, long max_visits) {
    AV mask = *(c.final_visit * c.valid);
    Univariate u = SecureUnivariate(c.visit_num, mask, 1, max_visits);
    std::vector<FreqRow> hist = SecureValueHistogram(c.visit_num, mask, 1, max_visits);
    if (party_id != 0) return;

    std::cout << "\n=== d1a  visits per patient  [" << ScopeName(c.scope)
              << ", final_visit = 1] ===" << std::endl;
    PrintUnivariate("visit_num", u);
    PrintFrequency("distribution", "visits_per_patient", hist);
    std::cout << "\n  NOTE: mean and sd are the wrong summary here -- visit counts are bounded\n"
                 "        below at 1 and heavily right skewed. The distribution above says more,\n"
                 "        in particular how many patients have exactly one visit: those patients\n"
                 "        contribute no within-patient information to any of the trend models.\n"
                 "  NOTE: quantiles use PCTLDEF=4 (linear interpolation). PROC UNIVARIATE\n"
                 "        defaults to PCTLDEF=5 and can differ by one observation's worth."
              << std::endl;
}

// --- node d1b: demographics at the index visit -------------------------------
void ReportD1b(const SecureCohort& c, int party_id) {
    AV mask = *(c.index_visit * c.valid);
    Univariate u = SecureUnivariate(c.newage, mask, 0, 120);
    std::vector<FreqRow> gender = SecureFrequency(c.gender_is, kGenderLevels, mask);
    std::vector<FreqRow> hispanic = SecureFrequency(c.hispanic_is, kHispanicLevels, mask);
    if (party_id != 0) return;

    std::cout << "\n=== d1b  demographics  [" << ScopeName(c.scope)
              << ", index_visit = 1] ===" << std::endl;
    PrintUnivariate("newage", u);
    PrintFrequency("gender", "gender", gender);
    for (size_t i = 0; i < gender.size(); ++i)
        std::cout << "      " << gender[i].value << " = " << kGenderNames[i] << std::endl;
    PrintFrequency("hispanic", "hispanic", hispanic);
    std::cout << "\n  NOTE: two one-way tables, not a crosstab -- the SAS asks for\n"
                 "        `table gender hispanic`, which is two separate tables.\n"
                 "  NOTE: check the hispanic cell counts here before trusting any model that\n"
                 "        adjusts for it. Sparsity here is what justifies dropping it from\n"
                 "        model 5b on the UMass subset."
              << std::endl;
}

// --- nodes sisa_perct_cnt{,_umass,_nonumass} ---------------------------------
struct SisaCounts {
    long window = 0;
    long pts_with_sisa = 0;
    long total_pts = 0;
    double sisa_pct = 0.0;
    long total_sa = 0;
    long null_fu_rows = 0;
};

std::vector<SisaCounts> ReportSisaCounts(SecureCohort& c, int party_id) {
    std::vector<SisaCounts> out;

    // Rows the follow-up windows cannot see at all.
    AV not_present = -c.fu_present;
    not_present += DataType(1);
    AV dropped = *(not_present * c.valid);
    const long null_rows = static_cast<long>(std::llround(OpenScalar(MaskCount(dropped), false)));

    for (DataType w : kFuWindows) {
        // in_window = fu_month is non-null AND equals w.
        //
        // Clone first: `AV a = b` is a SHALLOW copy that shares the underlying
        // buffer, so mutating it in place would corrupt the cohort column for
        // every later window.
        AV diff_lo = Clone(c.fu_month);
        diff_lo.setPrecision(0);
        diff_lo -= static_cast<DataType>(w) * DataType(scale);  // fu - w
        AV ge = *(diff_lo.gtez());
        AV diff_hi = -diff_lo;                                  // w - fu
        AV le = *(diff_hi.gtez());
        AV eq = *(le * ge);
        AV in_window = *(*(eq * c.fu_present) * c.valid);

        AV with_sisa = *(in_window * c.sisa);

        // Percentages count DISTINCT PATIENTS; the attempt total counts
        // ENCOUNTERS, with no DISTINCT, so a patient with two coded attempts in a
        // window contributes two. That asymmetry is in the source query and is
        // preserved deliberately.
        AV total_pts = CountDistinct(c.keys, in_window);
        AV pts_with_sisa = CountDistinct(c.keys, with_sisa);
        AV total_sa = MaskCount(*(in_window * c.sa));

        SisaCounts r;
        r.window = static_cast<long>(w);
        r.total_pts = static_cast<long>(std::llround(OpenScalar(total_pts, false)));
        r.pts_with_sisa = static_cast<long>(std::llround(OpenScalar(pts_with_sisa, false)));
        r.total_sa = static_cast<long>(std::llround(OpenScalar(total_sa, false)));
        r.sisa_pct = r.total_pts > 0 ? 100.0 * static_cast<double>(r.pts_with_sisa) /
                                           static_cast<double>(r.total_pts)
                                     : 0.0;
        r.null_fu_rows = null_rows;
        out.push_back(r);
    }

    if (party_id != 0) return out;

    const std::string suffix = c.scope == SystemScope::Any
                                   ? ""
                                   : (c.scope == SystemScope::UMass ? "_umass" : "_nonumass");
    std::cout << "\n=== sisa_perct_cnt" << suffix << "  [" << ScopeLabel(c.scope)
              << "] ===" << std::endl;
    std::cout << "\n  " << std::left << std::setw(10) << "window" << std::setw(18)
              << "pts_with_sisa" << std::setw(14) << "total_pts" << std::setw(12) << "sisa_pct"
              << std::setw(12) << "total_sa" << std::endl;
    for (const SisaCounts& r : out)
        std::cout << "  " << std::left << std::setw(10) << (std::to_string(r.window) + "m")
                  << std::setw(18) << r.pts_with_sisa << std::setw(14) << r.total_pts
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.sisa_pct
                  << std::defaultfloat << std::setw(12) << r.total_sa << std::endl;

    std::cout << "\n  NOTE: the denominator is not the cohort. total_pts counts patients with a\n"
                 "        record in the window, so sisa_pct is a prevalence conditional on being\n"
                 "        observed then -- not cumulative incidence by that month.\n"
                 "  NOTE: total_sa counts ENCOUNTERS, not patients (no DISTINCT in the source).\n"
                 "  NOTE: " << null_rows
              << " rows have a null fu_month and are invisible to every line above,\n"
                 "        appearing in neither numerator nor denominator and in no exclusion count."
              << std::endl;
    return out;
}


}  // namespace cdough::regression

#pragma once

#include <string>
#include <vector>

#include "./cohort.h"
#include "./primitives.h"

// The aggregate lineage nodes: the three sisa_perct_cnt tables.

namespace cdough::regression {

// Sum of a 0/1 mask: a count.
AV MaskCount(const AV& mask) {
    AV c = mask.chunkedSum(mask.size());
    c.setPrecision(0);
    return c;
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

// `reveal_to` names the party the counts belong to; -1 publishes them to every
// party, which is the two-owner default. The party that receives them is also
// the one that prints them, so a run with `--owner 0` leaves parties 1 and 2
// with nothing on stdout and nothing in their process memory.
//
// Collective: every party must call this, because the opens inside are.
std::vector<SisaCounts> ReportSisaCounts(SecureCohort& c, int party_id, int reveal_to = -1,
                                         bool print = true) {
    std::vector<SisaCounts> out;

    // Rows the follow-up windows cannot see at all.
    AV not_present = -c.fu_present;
    not_present += DataType(1);
    AV dropped = *(not_present * c.valid);
    const long null_rows = static_cast<long>(
        std::llround(OpenScalarToParty(MaskCount(dropped), reveal_to, party_id, false)));

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
        r.total_pts = static_cast<long>(
            std::llround(OpenScalarToParty(total_pts, reveal_to, party_id, false)));
        r.pts_with_sisa = static_cast<long>(
            std::llround(OpenScalarToParty(pts_with_sisa, reveal_to, party_id, false)));
        r.total_sa = static_cast<long>(
            std::llround(OpenScalarToParty(total_sa, reveal_to, party_id, false)));
        r.sisa_pct = r.total_pts > 0 ? 100.0 * static_cast<double>(r.pts_with_sisa) /
                                           static_cast<double>(r.total_pts)
                                     : 0.0;
        r.null_fu_rows = null_rows;
        out.push_back(r);
    }

    const int recipient = reveal_to < 0 ? 0 : reveal_to;
    if (party_id != recipient || !print) return out;

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

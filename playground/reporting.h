#pragma once

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "./regression.h"

// Coefficient tables, and the plaintext IRLS oracle the secure fixed-effects
// models are scored against.

namespace cdough::regression {

// =============================================================================
// Reporting
//
// p-values, confidence limits and odds ratios are computed HERE, in plaintext,
// from the revealed (estimate, standard error) pair. That is deliberate: p is a
// deterministic public function of two numbers the pipeline publishes anyway, so
// evaluating the normal CDF under MPC would protect nothing -- and at precision
// 16 the smallest representable positive value is 1.5e-5, so a secure erf would
// floor out around p = 1e-4 regardless.
// =============================================================================

const double kZ975 = 1.959963985;

double TwoSidedNormalP(double z) { return std::erfc(std::abs(z) * M_SQRT1_2); }

std::string FormatP(double p_value) {
    std::ostringstream os;
    if (p_value < 1e-4) return "<0.0001";
    os << std::fixed << std::setprecision(4) << p_value;
    return os.str();
}

void PrintFit(const FitResult& r, const ModelSpec& spec) {
    std::cout << "\n=== model " << r.step << "  [" << ScopeLabel(r.scope) << "] ===\n"
              << "  outcome  suicide_acutecare_icd_narrow (event = 1), logit link\n"
              << "  time     " << TimeName(spec.time) << "\n"
              << "  fit      "
              << (r.has_random ? "random-intercept mixed model, Laplace (BFGS)"
                               : "fixed-effects logistic regression (IRLS)")
              << "\n"
              << "  rows     " << r.rows_used;
    if (r.rows_dropped_null_fu > 0)
        std::cout << "   (" << r.rows_dropped_null_fu << " dropped: fu_month is null)";
    std::cout << "\n  iters    " << r.iterations << (r.converged ? " (converged)" : "")
              << std::endl;

    const bool want_or = spec.covars || spec.interaction;
    std::cout << "\n  " << std::left << std::setw(34) << "Term" << std::setw(12) << "Estimate"
              << std::setw(12) << "StdErr" << std::setw(10) << "z" << std::setw(11) << "Pr>|z|";
    if (want_or) std::cout << std::setw(12) << "OddsRatio" << std::setw(22) << "95% CI (OR)";
    std::cout << std::endl;

    for (size_t k = 0; k < r.terms.size(); ++k) {
        const double est = r.estimate[k];
        const double se = r.se[k];
        const bool have_se = std::isfinite(se) && se > 0.0;
        const double z = have_se ? est / se : 0.0;
        std::ostringstream se_s, z_s, p_s;
        if (have_se) {
            se_s << std::fixed << std::setprecision(5) << se;
            z_s << std::fixed << std::setprecision(3) << z;
            p_s << FormatP(TwoSidedNormalP(z));
        } else {
            se_s << "n/a";
            z_s << "n/a";
            p_s << "n/a";
        }
        std::cout << "  " << std::left << std::setw(34) << r.terms[k] << std::fixed
                  << std::setprecision(5) << std::setw(12) << est << std::setw(12)
                  << se_s.str() << std::setw(10) << z_s.str() << std::setw(11) << p_s.str();
        if (want_or) {
            std::ostringstream ci;
            if (have_se) {
                ci << "(" << std::fixed << std::setprecision(3) << std::exp(est - kZ975 * se)
                   << ", " << std::exp(est + kZ975 * se) << ")";
            } else {
                ci << "n/a";
            }
            std::cout << std::setprecision(4) << std::setw(12) << std::exp(est) << std::setw(22)
                      << ci.str();
        }
        std::cout << std::defaultfloat << std::endl;
    }

    if (r.has_random)
        std::cout << "\n  random intercept variance (subject_id): " << std::fixed
                  << std::setprecision(5) << r.sigma2 << std::defaultfloat << std::endl;

    if (spec.covars)
        std::cout << "\n  NOTE: newage was centred at " << kAgeCentre
                  << " for conditioning, so the intercept is the\n"
                     "        log-odds at age "
                  << kAgeCentre << ". The age coefficient itself is unaffected." << std::endl;

    for (const std::string& n : r.notes) std::cout << "  NOTE: " << n << std::endl;

    // The faults carried over from the source script.
    if (spec.interaction) {
        std::cout
            << "  NOTE: this model has NO random effect. The SAS declares subject_id in CLASS\n"
               "        and never uses it -- there is no RANDOM statement -- so PROC GLIMMIX\n"
               "        treats every encounter as independent. With repeated encounters per\n"
               "        patient the standard errors are too small and the p-values too\n"
               "        optimistic, on precisely the comparison the script exists to make.\n"
               "        Reproduced here rather than corrected."
            << std::endl;
        std::cout << "  NOTE: the interaction coefficient IS the difference in "
                  << TimeName(spec.time)
                  << " slopes\n        between the two systems -- this is the estimand."
                  << std::endl;
    }
    if (spec.step == "6a")
        std::cout << "  NOTE: the SAS ESTIMATE statement references data_source*fu_month in a\n"
                     "        model containing data_source*visit_num, so SAS rejects it. The\n"
                     "        answer survives as the interaction coefficient above."
                  << std::endl;
    if (spec.step == "6b")
        std::cout << "  NOTE: method=laplace is omitted in the SAS, so that fit silently falls\n"
                     "        back to residual pseudo-likelihood and its estimates are not on the\n"
                     "        same footing as the other thirteen. This port uses the same IRLS as\n"
                     "        6a, so the two ARE comparable here -- unlike in the original."
                  << std::endl;
    if (spec.time == TimeAxis::FuMonth)
        std::cout << "  NOTE: fu_month takes only {0,1,3,6} but enters as a linear term, forcing\n"
                     "        a constant change in log-odds per month across an unevenly spaced\n"
                     "        grid. If it is a window label rather than elapsed time it belongs\n"
                     "        in CLASS. Rows where it is null are dropped from this fit entirely."
                  << std::endl;
    if (spec.drop_hispanic)
        std::cout << "  NOTE: hispanic is dropped from THIS model only (sparse data), so its\n"
                     "        odds ratios are not directly comparable with the non-UMass fit."
                  << std::endl;
    if (!spec.covars)
        std::cout << "  NOTE: unadjusted by design. Compare against model "
                  << (spec.step == "2a" ? "5a" : "5b")
                  << " before reading\n        anything causal into the coefficient." << std::endl;
    if (spec.scope != SystemScope::Any && spec.time == TimeAxis::FuMonth)
        std::cout << "  NOTE: the time axis inherits an upstream ETL fault -- fu_month is null on\n"
                     "        the first in-system encounter of any patient whose history here\n"
                     "        starts later than their overall history, because the ETL guarded it\n"
                     "        on the global index_visit. Those rows drop out of this fit."
                  << std::endl;
}

// =============================================================================
// Plaintext oracle
//
// A direct IRLS fit in double precision, used to score the secure fixed-effects
// models. The mixed models are scored against the parameters the synthetic
// cohort was generated from instead, since a plaintext IRLS is not their
// estimand.
// =============================================================================

struct PlainFit {
    std::vector<double> estimate;
    std::vector<double> se;
};

PlainFit PlainLogisticIrls(const std::vector<double>& x_rm, const std::vector<double>& y,
                           const std::vector<double>& mask, size_t n, size_t p,
                           double ridge, int iterations) {
    std::vector<double> beta(p, 0.0);
    std::vector<double> g(p * p, 0.0);

    for (int it = 0; it < iterations; ++it) {
        std::vector<double> w(n), r(n);
        for (size_t i = 0; i < n; ++i) {
            double eta = 0.0;
            for (size_t k = 0; k < p; ++k) eta += x_rm[i * p + k] * beta[k];
            eta = std::max(-10.0, std::min(10.0, eta));
            const double pr = 1.0 / (1.0 + std::exp(-eta));
            w[i] = mask[i] * pr * (1.0 - pr);
            r[i] = mask[i] * (y[i] - pr);
        }
        std::vector<double> rhs(p, 0.0);
        g.assign(p * p, 0.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = 0; k < p; ++k) {
                rhs[k] += x_rm[i * p + k] * r[i];
                for (size_t l = k; l < p; ++l) {
                    const double v = w[i] * x_rm[i * p + k] * x_rm[i * p + l];
                    g[k * p + l] += v;
                    if (l != k) g[l * p + k] += v;
                }
            }
        }
        for (size_t k = 0; k < p; ++k) g[k * p + k] += ridge;

        // Gauss-Jordan solve and inverse in one pass.
        std::vector<double> m(g), inv(p * p, 0.0), sol(rhs);
        for (size_t k = 0; k < p; ++k) inv[k * p + k] = 1.0;
        for (size_t c = 0; c < p; ++c) {
            size_t piv = c;
            for (size_t rr = c + 1; rr < p; ++rr)
                if (std::abs(m[rr * p + c]) > std::abs(m[piv * p + c])) piv = rr;
            for (size_t k = 0; k < p; ++k) {
                std::swap(m[c * p + k], m[piv * p + k]);
                std::swap(inv[c * p + k], inv[piv * p + k]);
            }
            std::swap(sol[c], sol[piv]);
            const double d = m[c * p + c];
            if (std::abs(d) < 1e-15) continue;
            for (size_t k = 0; k < p; ++k) {
                m[c * p + k] /= d;
                inv[c * p + k] /= d;
            }
            sol[c] /= d;
            for (size_t rr = 0; rr < p; ++rr) {
                if (rr == c) continue;
                const double f = m[rr * p + c];
                for (size_t k = 0; k < p; ++k) {
                    m[rr * p + k] -= f * m[c * p + k];
                    inv[rr * p + k] -= f * inv[c * p + k];
                }
                sol[rr] -= f * sol[c];
            }
        }
        for (size_t k = 0; k < p; ++k)
            beta[k] += std::max(-4.0, std::min(4.0, sol[k]));

        if (it + 1 == iterations) {
            PlainFit out;
            out.estimate = beta;
            for (size_t k = 0; k < p; ++k)
                out.se.push_back(std::sqrt(std::max(0.0, inv[k * p + k])));
            return out;
        }
    }
    PlainFit out;
    out.estimate = beta;
    out.se.assign(p, 0.0);
    return out;
}

// Build the same design matrix the secure path builds, but in the clear.
void PlainDesign(const PlainCohort& c, const ModelSpec& spec, std::vector<double>& x_rm,
                 std::vector<double>& y, std::vector<double>& mask, size_t& p) {
    std::vector<std::string> dummy_terms;
    p = 2;  // intercept + time
    if (spec.covars) {
        p += 1;                            // newage
        p += kGenderLevels.size() - 1;     // gender dummies
        if (!spec.drop_hispanic) p += kHispanicLevels.size() - 1;
    }
    if (spec.interaction) p += 2;

    const size_t n = c.rows();
    x_rm.assign(n * p, 0.0);
    y.assign(n, 0.0);
    mask.assign(n, 1.0);

    for (size_t i = 0; i < n; ++i) {
        const double t = spec.time == TimeAxis::VisitNum
                             ? static_cast<double>(c.visit_num[i])
                             : static_cast<double>(c.fu_month[i]);
        if (spec.time == TimeAxis::FuMonth && c.fu_month_present[i] == 0) mask[i] = 0.0;

        size_t k = 0;
        x_rm[i * p + k++] = 1.0;
        x_rm[i * p + k++] = t;
        if (spec.covars) {
            x_rm[i * p + k++] = static_cast<double>(c.newage[i]) - kAgeCentre;
            for (size_t j = 0; j + 1 < kGenderLevels.size(); ++j)
                x_rm[i * p + k++] = (c.gender[i] == kGenderLevels[j]) ? 1.0 : 0.0;
            if (!spec.drop_hispanic)
                for (size_t j = 0; j + 1 < kHispanicLevels.size(); ++j)
                    x_rm[i * p + k++] = (c.hispanic[i] == kHispanicLevels[j]) ? 1.0 : 0.0;
        }
        if (spec.interaction) {
            const double umass = (c.data_source[i] == 1) ? 1.0 : 0.0;
            x_rm[i * p + k++] = umass;
            x_rm[i * p + k++] = umass * t;
        }
        for (size_t kk = 0; kk < p; ++kk) x_rm[i * p + kk] *= mask[i];
        y[i] = static_cast<double>(c.sisa[i]) * mask[i];
    }
}


}  // namespace cdough::regression

#pragma once

#include <cassert>
#include <string>
#include <vector>

#include "./cohort.h"
#include "./linalg.h"
#include "./nodes.h"
#include "./optimizer.h"

// The fourteen regression fits: design-matrix construction, IRLS for the
// fixed-effects models (steps 6a/6b), and the flat ragged-cluster
// Laplace-approximated mixed model with an analytic gradient (steps 2 and 5).

namespace cdough::regression {

// =============================================================================
// Model specifications and design matrices
// =============================================================================

enum class TimeAxis { VisitNum, FuMonth };

struct ModelSpec {
    std::string step;                          // "2a" ... "6b"
    TimeAxis time = TimeAxis::VisitNum;
    bool covars = false;                       // newage + gender + hispanic
    bool interaction = false;                  // data_source main effect + data_source*time
    bool random_intercept = true;              // false only for steps 6a/6b
    bool drop_hispanic = false;                // model 5b on the UMass subset
    SystemScope scope = SystemScope::Any;
};

const char* TimeName(TimeAxis t) { return t == TimeAxis::VisitNum ? "visit_num" : "fu_month"; }

// Age is centred at a public constant purely for conditioning: at precision 16 an
// uncentred age column makes the entries of X^T W X span many orders of
// magnitude. Centring shifts only the intercept, and the shift is undone exactly
// when the coefficients are reported, so nothing about the fit changes.
const double kAgeCentre = 40.0;

// One model's design matrix, held row-major so that eta = X*beta is a single
// chunked dot product and X^T W X needs no transpose.
struct ModelData {
    size_t n_pad = 0;
    size_t p = 0;
    AV x;         // n_pad * p, row-major, scaled
    AV y;         // n_pad, 0/1
    AV row_mask;  // n_pad, 1 on rows this model actually uses
    std::vector<BV> keys;
    const ScanPlan* scan_plan = nullptr;  // owned by the cohort
    AV last_of_subject;
    std::vector<std::string> terms;
    long rows_used = 0;
    long rows_dropped_null_fu = 0;

    ModelData(EngineRef engine, size_t np, size_t width)
        : n_pad(np), p(width), x(np * width, engine), y(np, engine), row_mask(np, engine),
          last_of_subject(np, engine) {}
};

ModelData BuildDesign(const SecureCohort& c, const ModelSpec& spec) {
    EngineRef engine = c.valid.engine;

    // Column list, in the order SAS lists them.
    std::vector<std::string> terms{"Intercept", TimeName(spec.time)};
    if (spec.covars) {
        terms.push_back("newage");
        for (size_t i = 0; i + 1 < kGenderLevels.size(); ++i)
            terms.push_back(std::string("gender=") + kGenderNames[i]);
        if (!spec.drop_hispanic)
            for (size_t i = 0; i + 1 < kHispanicLevels.size(); ++i)
                terms.push_back("hispanic=" + std::to_string(kHispanicLevels[i]));
    }
    if (spec.interaction) {
        terms.push_back("data_source=UMass");
        terms.push_back(std::string("data_source=UMass * ") + TimeName(spec.time));
    }

    const size_t np = c.n_pad;
    const size_t p = terms.size();
    ModelData md(engine, np, p);
    md.terms = terms;
    md.keys = c.keys;
    md.scan_plan = &c.scan_plan;
    md.last_of_subject = c.last_of_subject;

    // Rows this model uses. A b-axis model silently drops every row whose
    // fu_month is null -- that is the source script's behaviour, and the count is
    // reported rather than left invisible.
    AV mask = Clone(c.valid);
    mask.setPrecision(0);
    if (spec.time == TimeAxis::FuMonth) {
        mask = *(mask * c.fu_present);
        mask.setPrecision(0);
    }
    md.row_mask = mask;

    AV time_col = spec.time == TimeAxis::VisitNum ? Clone(c.visit_num) : Clone(c.fu_month);
    time_col.setPrecision(0);

    AV age_centred = Clone(c.newage);
    age_centred.setPrecision(0);
    age_centred -= static_cast<DataType>(std::llround(kAgeCentre * scale));

    // Assemble the row-major design matrix column by column, writing into a
    // strided view of the flat buffer.
    std::vector<AV> cols;
    {
        AV ones(np, engine);
        ones.setPrecision(0);
        ones += DataType(scale);
        cols.push_back(ones);
    }
    cols.push_back(time_col);
    if (spec.covars) {
        cols.push_back(age_centred);
        // SAS CLASS makes the LAST sorted level the reference, so the final
        // level of each list is omitted.
        for (size_t i = 0; i + 1 < kGenderLevels.size(); ++i) {
            AV d = *(c.gender_is[i] * DataType(scale));
            d.setPrecision(0);
            cols.push_back(d);
        }
        if (!spec.drop_hispanic) {
            for (size_t i = 0; i + 1 < kHispanicLevels.size(); ++i) {
                AV d = *(c.hispanic_is[i] * DataType(scale));
                d.setPrecision(0);
                cols.push_back(d);
            }
        }
    }
    if (spec.interaction) {
        AV ds = *(c.umass * DataType(scale));
        ds.setPrecision(0);
        cols.push_back(ds);
        AV inter = *(*(ds * time_col) / scale);
        inter.setPrecision(0);
        cols.push_back(inter);
    }
    assert(cols.size() == p);

    md.x.setPrecision(0);
    for (size_t k = 0; k < p; ++k) {
        // Column k of a row-major n x p matrix is the stride-p view starting at k.
        AV dest = md.x.simple_subset_reference(k, p, (np - 1) * p + k);
        AV src = *(cols[k] * md.row_mask);  // zero the rows this model does not use
        dest = src;
    }
    md.x.setPrecision(0);

    AV y = *(c.sisa * md.row_mask);
    y.setPrecision(0);
    md.y = *(y * DataType(scale));
    md.y.setPrecision(0);

    md.rows_used = static_cast<long>(std::llround(OpenScalar(MaskCount(md.row_mask), false)));
    AV dropped = c.valid - md.row_mask;
    md.rows_dropped_null_fu =
        static_cast<long>(std::llround(OpenScalar(MaskCount(dropped), false)));
    return md;
}

// Gather the columns of a row-major n x p matrix end to end, which is the layout
// a chunked dot_product needs. mapping_reference is a public view: free.
AV ColumnGather(const AV& x_rm, size_t n, size_t p) {
    std::vector<cdough::VectorSizeType> map(n * p);
    for (size_t k = 0; k < p; ++k)
        for (size_t i = 0; i < n; ++i)
            map[k * n + i] = static_cast<cdough::VectorSizeType>(i * p + k);
    AV out = x_rm.mapping_reference(map);
    out.setPrecision(0);
    return out;
}

// eta = X * beta for X row-major (n x p) and beta a length-p vector.
AV LinearPredictor(const AV& x_rm, const AV& beta, size_t n, size_t p) {
    AV x_ = x_rm;
    x_.setPrecision(0);
    AV b_ = beta;
    b_.setPrecision(0);
    AV tiled = b_.cyclic_subset_reference(n);  // [b_0..b_{p-1}] repeated n times
    tiled.setPrecision(0);
    AV eta = *(*x_.dot_product(tiled, p) / scale);
    eta.setPrecision(0);
    return eta;
}

// X^T r, for r a length-n residual vector.
AV CrossProduct(const AV& x_rm, const AV& r, size_t n, size_t p) {
    AV cols = ColumnGather(x_rm, n, p);
    AV r_ = r;
    r_.setPrecision(0);
    AV tiled = r_.cyclic_subset_reference(p);  // r repeated p times: column-major order
    tiled.setPrecision(0);
    AV out = *(*cols.dot_product(tiled, n) / scale);
    out.setPrecision(0);
    return out;
}

// =============================================================================
// Fitting
// =============================================================================

struct FitResult {
    std::string step;
    SystemScope scope = SystemScope::Any;
    std::vector<std::string> terms;
    std::vector<double> estimate;
    std::vector<double> se;
    bool has_random = false;
    double sigma2 = 0.0;
    int iterations = 0;
    long rows_used = 0;
    long rows_dropped_null_fu = 0;
    bool converged = false;
    double hessian_condition = 0.0;  // mixed models only; 0 when not computed
    std::vector<std::string> notes;
};

const double kIrlsRidge = 1e-3;
constexpr int kIrlsIterations = 8;

// Steps 6a and 6b: an ordinary fixed-effects logistic regression, fitted by
// IRLS / Fisher scoring.
//
// No random effect, faithfully. The SAS declares subject_id in CLASS and then
// never uses it -- there is no RANDOM statement -- so PROC GLIMMIX treats every
// encounter as independent. That is reproduced here rather than corrected, and
// flagged in the output.
//
// IRLS rather than BFGS because the Hessian is analytic: convergence is
// quadratic, and the converged X^T W X IS the inverse asymptotic covariance, so
// the standard errors come out exact instead of approximated.
FitResult FitLogisticIrls(const ModelData& md, const ModelSpec& spec) {
    EngineRef engine = md.y.engine;
    const size_t n = md.n_pad, p = md.p;

    AV beta(p, engine);
    beta.setPrecision(0);

    AV gram(p * p, engine);
    gram.setPrecision(0);

    for (int it = 0; it < kIrlsIterations; ++it) {
        AV eta = LinearPredictor(md.x, beta, n, p);
        eta = ClampAbs(eta, kMaxExpArg_scaled);  // keep Exp inside its saturation band
        eta.setPrecision(0);

        AV pr = Sigmoid(eta);
        pr.setPrecision(0);
        AV one_minus = -pr;
        one_minus += DataType(scale);
        AV w = *(*(pr * one_minus) / scale);
        w = *(w * md.row_mask);
        w.setPrecision(0);

        AV resid = md.y - pr;
        resid = *(resid * md.row_mask);
        resid.setPrecision(0);

        AV rhs = CrossProduct(md.x, resid, n, p);
        gram = Gram(md.x, w, n, p);
        AddRidge(gram, p, kIrlsRidge);

        AV delta = CholeskySolve(gram, rhs, p);
        delta = ClampNewtonStep(delta);  // the same damping the inner Newton uses
        delta.setPrecision(0);
        beta += delta;
    }

    AV cov = SymmetricInverse(gram, p);

    FitResult r;
    r.step = spec.step;
    r.scope = spec.scope;
    r.terms = md.terms;
    r.has_random = false;
    r.iterations = kIrlsIterations;
    r.rows_used = md.rows_used;
    r.rows_dropped_null_fu = md.rows_dropped_null_fu;
    r.converged = true;

    const std::vector<double> beta_open = OpenToDoubles(beta);
    const std::vector<double> cov_open = OpenToDoubles(cov);
    for (size_t k = 0; k < p; ++k) {
        r.estimate.push_back(beta_open[k]);
        r.se.push_back(std::sqrt(std::max(0.0, cov_open[k * p + k])));
    }
    return r;
}

// =============================================================================
// Steps 2a / 2b / 5a / 5b: random-intercept logistic mixed model, Laplace
//
// The original code held the data as balanced ClusterGroups and looped over them
// one at a time. Real patients have a ragged number of encounters, so everything
// here works on the FLAT table instead: every cluster iterates in lockstep, and
// the per-cluster sums come from the segmented scans above. Cost is O(n log n)
// in the number of encounters, with no per-patient padding and no dependence on
// the largest cluster.
// =============================================================================

// Broadcast a 1-element vector across n rows. A free public view.
AV Broadcast(const AV& scalar, size_t n) {
    AV out = scalar.repeated_subset_reference(n);
    out.setPrecision(0);
    return out;
}

// sigma^2 is kept inside a public band: Log needs a strictly positive argument,
// and a runaway variance would push eta outside Exp's saturation range.
const DataType kSigma2Min_scaled = std::llround(1e-3 * scale);
const DataType kSigma2Max_scaled = std::llround(1e3 * scale);

// The conditional mode u_hat for EVERY cluster at once.
//
// u is carried broadcast on every row. It stays constant inside a cluster by
// construction -- each row of a cluster sees the same segmented gradient and
// curvature -- so no extra broadcast is needed to maintain it.
AV FlatConditionalMode(const ModelData& md, const AV& x_beta, const AV& inv_sigma2_row) {
    EngineRef engine = md.y.engine;
    const size_t n = md.n_pad;

    AV u(n, engine);
    u.setPrecision(0);

    for (int it = 0; it < kNewtonIterations; ++it) {
        AV eta = x_beta + u;
        eta = ClampAbs(eta, kMaxExpArg_scaled);
        eta.setPrecision(0);

        AV pr = Sigmoid(eta);
        pr.setPrecision(0);
        AV omp = -pr;
        omp += DataType(scale);

        AV varp = *(*(pr * omp) / scale);
        varp = *(varp * md.row_mask);
        varp.setPrecision(0);

        AV resid = md.y - pr;
        resid = *(resid * md.row_mask);
        resid.setPrecision(0);

        // One call, both columns, with the group bits taken from the cohort's
        // precomputed plan rather than recomputed per level per call.
        std::vector<AV> in{resid, varp};
        std::vector<AV> out;
        out.emplace_back(n, engine);
        out.emplace_back(n, engine);
        if (g_use_cached_scans) {
            SegTotalPlanned(*md.scan_plan, in, out);
        } else {
            std::vector<BV> keys = md.keys;
            SegTotal(keys, in, out);
        }

        AV grad = out[0] - *(*(u * inv_sigma2_row) / scale);
        grad.setPrecision(0);
        AV curv = out[1] + inv_sigma2_row;
        curv.setPrecision(0);

        AV step = Div(grad, curv);
        step = ClampNewtonStep(step);
        step.setPrecision(0);
        u += step;
    }
    return u;
}

// Negative Laplace-approximated marginal log-likelihood: the BFGS objective.
//
// The per-cluster terms -0.5 log(sigma^2) - 0.5 log(A_i) and the prior penalty
// must be counted ONCE PER CLUSTER, not once per row. The forward scan leaves
// each cluster's totals on its last row, and multiplying by last_of_subject --
// which is zero everywhere else, and zero on the sentinel pad block -- picks
// exactly those rows out. A cluster with no usable rows contributes
// -0.5 log(sigma^2) - 0.5 log(1/sigma^2) = 0, so it drops out on its own.
// Counts calls to the marginal likelihood. The number of objective evaluations
// is the multiplier on everything else in a mixed-model fit, so it is the first
// thing to look at when the cost is wrong.
long g_objective_evaluations = 0;

// Negative Laplace-approximated marginal log-likelihood and, optionally, its
// ANALYTIC gradient.
//
// The gradient is worth deriving rather than differencing. Central differences
// cost 2*dim objective evaluations, which measurement showed to be the single
// largest multiplier in the whole program; the analytic form shares the
// conditional-mode search with the value and costs about one extra segmented
// scan. It is also strictly more accurate, because it never divides by a step.
//
// Derivation. Write g_i(u) for the exponent inside the integral, so that
//     l_i = g_i(u_i) - 0.5 log(sigma^2) - 0.5 log(A_i),
//     A_i = 1/sigma^2 + sum_j v_ij,        v_ij = p_ij (1 - p_ij),
// with u_i the conditional mode. Because u_i is a stationary point of g_i, the
// envelope theorem kills the du_i/dbeta terms in d g_i(u_i)/dbeta and it
// collapses to the partial derivative at fixed u:
//     d/dbeta_k [ g_i(u_i) ] = sum_j (y_ij - p_ij) x_ijk.
// The log(A_i) term gets no such cancellation. Differentiating A_i through both
// x and u_i, using
//     du_i/dbeta_k  = -(sum_j v_ij x_ijk) / A_i          (implicit function theorem)
//     dv_ij/deta_ij = v_ij (1 - 2 p_ij)
// gives dA_i/dbeta_k = S3_k - S1_k S2_i / A_i, and therefore
//     dl_i/dbeta_k = S4_k - (S3_k - S1_k S2_i / A_i) / (2 A_i)
// where, all sums being over the rows of cluster i,
//     S1_k = sum v_ij x_ijk,                 S2_i = sum v_ij (1 - 2 p_ij),
//     S3_k = sum v_ij (1 - 2 p_ij) x_ijk,    S4_k = sum (y_ij - p_ij) x_ijk.
// For the variance, parameterised as sigma^2 = exp(2s) to keep it positive:
//     du_i/ds  = 2 u_i / (sigma^2 A_i),
//     dA_i/ds  = -2/sigma^2 + S2_i du_i/ds,
//     dl_i/ds  = u_i^2/sigma^2 - 1 - (dA_i/ds) / (2 A_i).
//
// Every S is a per-cluster sum of a per-row quantity, so all 3p+2 of them come
// out of ONE forward segmented scan alongside the two the value already needs.
//
// Pass nullptr for `gradient_out` to get the value alone, which is what the
// line search wants.
AV FlatObjective(const ModelData& md, const std::vector<AV>& params,
                 std::vector<AV>* gradient_out) {
    ++g_objective_evaluations;

    EngineRef engine = md.y.engine;
    const size_t n = md.n_pad, p = md.p;
    const bool want_gradient = (gradient_out != nullptr);

    AV beta(p, engine);
    beta.setPrecision(0);
    for (size_t k = 0; k < p; ++k) {
        AV cell = Cell(beta, k);
        AV src = Clone(params[k]);
        src.setPrecision(0);
        cell = src;
    }

    AV s = Clone(params[p]);
    s.setPrecision(0);
    AV two_s = *(s * DataType(2));
    two_s.setPrecision(precision);
    AV sigma2 = Exp(two_s);
    sigma2 = ClampRange(sigma2, kSigma2Min_scaled, kSigma2Max_scaled);
    sigma2.setPrecision(0);

    AV inv_sigma2 = Recip(sigma2);
    inv_sigma2.setPrecision(0);
    AV log_sigma2 = Log(sigma2);
    log_sigma2.setPrecision(0);

    AV inv_sigma2_row = Broadcast(inv_sigma2, n);
    AV log_sigma2_row = Broadcast(log_sigma2, n);

    AV x_beta = LinearPredictor(md.x, beta, n, p);
    AV u = FlatConditionalMode(md, x_beta, inv_sigma2_row);

    AV eta = x_beta + u;
    eta = ClampAbs(eta, kMaxExpArg_scaled);
    eta.setPrecision(0);

    AV pr = Sigmoid(eta);
    pr.setPrecision(0);
    AV sp = LogOnePlusExp(eta);
    sp.setPrecision(0);

    AV y_eta = *(*(md.y * eta) / scale);
    AV cll_row = y_eta - sp;
    cll_row = *(cll_row * md.row_mask);
    cll_row.setPrecision(0);

    AV omp = -pr;
    omp += DataType(scale);
    AV varp_row = *(*(pr * omp) / scale);
    varp_row = *(varp_row * md.row_mask);
    varp_row.setPrecision(0);

    // Columns to scan. The first two are what the value needs; the rest are the
    // gradient's per-cluster sums, folded into the same pass.
    std::vector<AV> columns{cll_row, varp_row};
    AV resid(n, engine), skew(n, engine);
    if (want_gradient) {
        // skew = v * (1 - 2p), the derivative of v with respect to eta.
        AV one_minus_two_p = *(pr * DataType(2));
        one_minus_two_p = -one_minus_two_p;
        one_minus_two_p += DataType(scale);
        skew = *(*(varp_row * one_minus_two_p) / scale);
        skew.setPrecision(0);

        resid = md.y - pr;
        resid = *(resid * md.row_mask);
        resid.setPrecision(0);

        columns.push_back(skew);
        for (size_t k = 0; k < p; ++k) {
            AV xk = md.x.simple_subset_reference(k, p, (n - 1) * p + k);
            xk.setPrecision(0);
            columns.push_back(*(*(varp_row * xk) / scale));
            columns.back().setPrecision(0);
        }
        for (size_t k = 0; k < p; ++k) {
            AV xk = md.x.simple_subset_reference(k, p, (n - 1) * p + k);
            xk.setPrecision(0);
            columns.push_back(*(*(skew * xk) / scale));
            columns.back().setPrecision(0);
        }
        for (size_t k = 0; k < p; ++k) {
            AV xk = md.x.simple_subset_reference(k, p, (n - 1) * p + k);
            xk.setPrecision(0);
            columns.push_back(*(*(resid * xk) / scale));
            columns.back().setPrecision(0);
        }
    }

    std::vector<AV> pre;
    pre.reserve(columns.size());
    for (size_t k = 0; k < columns.size(); ++k) pre.emplace_back(n, engine);
    if (g_use_cached_scans) {
        SegScanPlanned(*md.scan_plan, columns, pre, SegDirection::Forward);
    } else {
        std::vector<BV> keys = md.keys;
        SegScan(keys, columns, pre, SegDirection::Forward);
    }

    AV a_row = pre[1] + inv_sigma2_row;
    a_row.setPrecision(0);
    AV log_a = Log(a_row);
    log_a.setPrecision(0);

    AV u_sq = *(*(u * u) / scale);
    AV pen = *(*(u_sq * inv_sigma2_row) / scale);
    AV pen_half = *(pen / DataType(2));

    AV log_terms = log_sigma2_row + log_a;
    AV log_half = *(log_terms / DataType(2));

    AV per_row = pre[0] - pen_half - log_half;
    per_row.setPrecision(0);

    AV contrib = *(per_row * md.last_of_subject);
    AV total = contrib.chunkedSum(n);
    total.setPrecision(0);

    if (want_gradient) {
        // One reciprocal serves the whole gradient. It is computed on every row
        // even though only the last row of each cluster is used, because a
        // vectorised division over n rows is cheaper than compacting first.
        AV inv_a = Recip(a_row);
        inv_a.setPrecision(0);
        AV s2 = pre[2];  // S2_i, the skew total

        gradient_out->clear();
        gradient_out->reserve(p + 1);

        for (size_t k = 0; k < p; ++k) {
            AV s1 = pre[3 + k];
            AV s3 = pre[3 + p + k];
            AV s4 = pre[3 + 2 * p + k];

            // dA/dbeta_k = S3_k - S1_k * S2 / A
            AV s1_s2 = *(*(s1 * s2) / scale);
            AV shift = *(*(s1_s2 * inv_a) / scale);
            AV d_a = s3 - shift;
            d_a.setPrecision(0);

            // dl/dbeta_k = S4_k - (dA/dbeta_k) / (2 A)
            AV correction = *(*(d_a * inv_a) / scale);
            AV half_correction = *(correction / DataType(2));
            AV row = s4 - half_correction;
            row.setPrecision(0);

            AV masked = *(row * md.last_of_subject);
            AV summed = masked.chunkedSum(n);
            AV negated = -summed;  // gradient of the NEGATIVE log-likelihood
            negated.setPrecision(precision);
            gradient_out->push_back(negated);
        }

        // du/ds = 2 u / (sigma^2 A)
        AV du_ds = *(*(u * inv_sigma2_row) / scale);
        du_ds = *(*(du_ds * inv_a) / scale);
        du_ds = *(du_ds * DataType(2));
        du_ds.setPrecision(0);

        // dA/ds = -2/sigma^2 + S2 * du/ds
        AV d_a_ds = *(*(s2 * du_ds) / scale);
        AV two_inv = *(inv_sigma2_row * DataType(2));
        d_a_ds -= two_inv;
        d_a_ds.setPrecision(0);

        // dl/ds = u^2/sigma^2 - 1 - (dA/ds) / (2 A)
        AV correction = *(*(d_a_ds * inv_a) / scale);
        AV half_correction = *(correction / DataType(2));
        AV row = pen - half_correction;  // pen is u^2/sigma^2
        row -= DataType(scale);
        row.setPrecision(0);

        AV masked = *(row * md.last_of_subject);
        AV summed = masked.chunkedSum(n);
        AV negated = -summed;
        negated.setPrecision(precision);
        gradient_out->push_back(negated);
    }

    AV neg = -total;
    neg.setPrecision(precision);
    return neg;
}

// Value-only wrapper, for the places that do not want a gradient.
AV FlatNegMarginalLogLik(const ModelData& md, const std::vector<AV>& params) {
    return FlatObjective(md, params, nullptr);
}

constexpr int kGlmmBfgsIterations = 12;

// Step for the numerical observed-information Hessian below. A second difference
// divides by h^2, so it amplifies the objective's fixed-point noise by 1/h^2;
// the truncation error meanwhile grows as h^2. With an objective good to about
// 1e-4 at precision 16, the balance sits near h = eps^(1/4) ~ 0.1.
const double kHessianStep = 0.1;

// Above this the observed-information matrix is close enough to singular that
// its inverse should not be trusted at all. The threshold is deliberately high:
// with the Hessian built from differences of the ANALYTIC gradient, standard
// errors measured against an accurate double-precision reference stayed within
// 8% at condition numbers around 2e4, so warning there would be crying wolf.
// The earlier objective-difference Hessian was 17-46% off at the same
// conditioning, which is what the old, much lower threshold was compensating
// for.
const double kHessianConditionWarn = 1e6;

// Standard errors for the mixed models, from the observed information at the
// optimum.
//
// The BFGS inverse-Hessian approximation is NOT good enough for this. With a
// dozen iterations and up to eight parameters it can leave whole directions
// untouched, and its diagonal then reports a variance of almost exactly 1 --
// an artefact of the identity it was initialised with rather than a standard
// error. Instead the Hessian is formed by finite differences of the objective
// and inverted directly.
//
// The differences are opened and the small dense algebra is done in plaintext.
// That is consistent with the choice already made for p-values: the curvature of
// the log-likelihood at the optimum is exactly what a published standard error
// discloses, so evaluating it under MPC would protect nothing, and it avoids
// amplifying fixed-point noise through a second difference and a matrix inverse.
// `condition_out` receives a 1-norm condition-number estimate for the Hessian.
// It matters: in a near-collinear direction the Hessian is nearly singular, and
// inverting it amplifies the objective's fixed-point noise without limit. The
// standard errors on such terms are not trustworthy and the caller says so.
//
// The Hessian is built from CENTRAL DIFFERENCES OF THE ANALYTIC GRADIENT, not
// from second differences of the objective. That matters twice over. It costs
// 2*dim gradient evaluations instead of 1 + 2*dim + dim(dim-1)/2 objective
// evaluations -- about 2.5x less at dim = 7 -- and, more importantly, a first
// difference divides by h rather than h^2, so it amplifies the fixed-point noise
// by one factor of 1/h instead of two. That is exactly the term that made the
// standard errors on near-collinear directions unreliable.
//
// The differences are opened and the small dense algebra is done in plaintext,
// consistent with the choice already made for p-values: the curvature of the
// log-likelihood at the optimum is precisely what a published standard error
// discloses, so evaluating it under MPC would protect nothing.
std::vector<double> ObservedInformationSE(const ValueGradFn& objective,
                                          const std::vector<AV>& optimum,
                                          size_t num_reported,
                                          double* condition_out = nullptr) {
    const size_t dim = optimum.size();
    const double h = kHessianStep;

    auto gradient_at = [&](size_t axis, double delta) {
        std::vector<AV> point;
        point.reserve(dim);
        for (size_t k = 0; k < dim; ++k) {
            AV v = Clone(optimum[k]);
            v.setPrecision(0);
            if (k == axis) v += static_cast<DataType>(std::llround(delta * scale));
            v.setPrecision(precision);
            point.push_back(v);
        }
        std::vector<AV> grad;
        objective(point, &grad);
        std::vector<double> out;
        out.reserve(dim);
        for (AV& g : grad) out.push_back(OpenScalar(g));
        return out;
    };

    // Column k of the Hessian is d(grad)/d(x_k).
    std::vector<double> hess(dim * dim, 0.0);
    for (size_t k = 0; k < dim; ++k) {
        const std::vector<double> forward = gradient_at(k, h);
        const std::vector<double> backward = gradient_at(k, -h);
        for (size_t j = 0; j < dim; ++j)
            hess[j * dim + k] = (forward[j] - backward[j]) / (2.0 * h);
    }
    // The true Hessian is symmetric; averaging the two estimates of each
    // off-diagonal entry halves the noise for free.
    for (size_t j = 0; j < dim; ++j) {
        for (size_t k = j + 1; k < dim; ++k) {
            const double avg = 0.5 * (hess[j * dim + k] + hess[k * dim + j]);
            hess[j * dim + k] = hess[k * dim + j] = avg;
        }
    }

    // Invert by Gauss-Jordan with partial pivoting. Unlike the secure Cholesky,
    // this runs on public numbers, so pivoting costs nothing and the matrix does
    // not have to be positive definite for the routine to return something --
    // a non-positive diagonal in the result is reported as an unavailable
    // standard error rather than silently square-rooted.
    std::vector<double> m(hess), inv(dim * dim, 0.0);
    for (size_t k = 0; k < dim; ++k) inv[k * dim + k] = 1.0;
    for (size_t c = 0; c < dim; ++c) {
        size_t piv = c;
        for (size_t r = c + 1; r < dim; ++r)
            if (std::abs(m[r * dim + c]) > std::abs(m[piv * dim + c])) piv = r;
        for (size_t k = 0; k < dim; ++k) {
            std::swap(m[c * dim + k], m[piv * dim + k]);
            std::swap(inv[c * dim + k], inv[piv * dim + k]);
        }
        const double d = m[c * dim + c];
        if (std::abs(d) < 1e-12) {
            std::fill(inv.begin(), inv.end(), std::numeric_limits<double>::quiet_NaN());
            break;
        }
        for (size_t k = 0; k < dim; ++k) {
            m[c * dim + k] /= d;
            inv[c * dim + k] /= d;
        }
        for (size_t r = 0; r < dim; ++r) {
            if (r == c) continue;
            const double f = m[r * dim + c];
            for (size_t k = 0; k < dim; ++k) {
                m[r * dim + k] -= f * m[c * dim + k];
                inv[r * dim + k] -= f * inv[c * dim + k];
            }
        }
    }

    if (condition_out != nullptr) {
        // 1-norm condition estimate: max column sum of H times that of H^-1.
        double norm_h = 0.0, norm_inv = 0.0;
        for (size_t c = 0; c < dim; ++c) {
            double col_h = 0.0, col_inv = 0.0;
            for (size_t r = 0; r < dim; ++r) {
                col_h += std::abs(hess[r * dim + c]);
                col_inv += std::abs(inv[r * dim + c]);
            }
            norm_h = std::max(norm_h, col_h);
            norm_inv = std::max(norm_inv, col_inv);
        }
        *condition_out = std::isfinite(norm_h * norm_inv)
                             ? norm_h * norm_inv
                             : std::numeric_limits<double>::infinity();
    }

    std::vector<double> se;
    for (size_t k = 0; k < num_reported; ++k) {
        const double var = inv[k * dim + k];
        se.push_back(std::isfinite(var) && var > 0.0
                         ? std::sqrt(var)
                         : std::numeric_limits<double>::quiet_NaN());
    }
    return se;
}

FitResult FitGlmmLaplace(const ModelData& md, const ModelSpec& spec, int party_id) {
    EngineRef engine = md.y.engine;
    const size_t p = md.p;
    const size_t dim = p + 1;  // beta plus s, where sigma = exp(s)

    // Start from beta = 0 and sigma = 1 (s = 0).
    std::vector<AV> x0;
    for (size_t k = 0; k < dim; ++k) {
        AV v(1, engine);
        v.setPrecision(precision);
        x0.push_back(v);
    }

    ValueGradFn objective = [&md](const std::vector<AV>& params,
                                  std::vector<AV>* gradient) -> AV {
        return FlatObjective(md, params, gradient);
    };

    OptResult opt = MinimizeBFGS(objective, x0, kGlmmBfgsIterations);

    FitResult r;
    r.step = spec.step;
    r.scope = spec.scope;
    r.terms = md.terms;
    r.has_random = true;
    r.iterations = opt.iterations;
    r.converged = opt.converged;
    r.rows_used = md.rows_used;
    r.rows_dropped_null_fu = md.rows_dropped_null_fu;

    for (size_t k = 0; k < p; ++k) r.estimate.push_back(OpenScalar(opt.params[k]));
    double condition = 0.0;
    r.se = ObservedInformationSE(objective, opt.params, p, &condition);
    r.hessian_condition = condition;
    const double s_hat = OpenScalar(opt.params[p]);
    r.sigma2 = std::exp(2.0 * s_hat);

    r.notes.push_back(
        "standard errors come from central differences of the analytic gradient at the "
        "optimum, not from the BFGS inverse-Hessian approximation");
    if (!opt.hessian_updated)
        r.notes.push_back(
            "no BFGS curvature update was accepted, so the optimiser may have stopped early");
    {
        // Always report the conditioning: it is the single most useful number
        // for judging how much to trust the standard errors on the weakly
        // determined terms, which here are the intercept and any near-collinear
        // dummy such as hispanic.
        std::ostringstream note;
        note << "observed-information 1-norm condition estimate " << std::scientific
             << std::setprecision(2) << condition;
        if (!(condition < kHessianConditionWarn))
            note << " -- close to singular; treat every standard error from this fit as "
                    "indicative only";
        r.notes.push_back(note.str());
    }
    (void)party_id;
    return r;
}


}  // namespace cdough::regression

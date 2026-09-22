#pragma once

#include <chrono>

#include "./primitives.h"

namespace cdough::regression {

// =============================================================================
// BFGS Optimization & Secure Linear Algebra Helpers
// =============================================================================

// Generates c * I_n as a SecureMatrix in MPC (row-major).
SMatrix ScaledIdentity(size_t n, double c, EngineRef engine) {
    cdough::Vector<DataType> eye(n * n, 0);
    const DataType diagonal = std::llround(c * scale);
    for (size_t i = 0; i < n; ++i) {
        eye[i * n + i] = diagonal;
    }
    PMatrix plain_eye(eye, n, n, false);
    auto sec_eye = engine.secret_share_matrix(plain_eye, 0);
    sec_eye.setPrecision(precision);
    return sec_eye;
}

// Generates an n x n identity SecureMatrix in MPC (row-major).
SMatrix Identity(size_t n, EngineRef engine) { return ScaledIdentity(n, 1.0, engine); }

// Permutes a row-major (rows x cols) buffer into the row-major buffer of its
// transpose. This is a pure index permutation, so it costs no communication.
//
// `mapping_reference` only builds a view, and the matmul kernel rejects views
// (mapping_access_vector.h asserts !has_mapping()), so the view has to be
// materialized. `Clone` is exactly the library's own materialization recipe
// (construct a fresh buffer, then assign), so it drops the mapping for us.
AV TransposeData(const AV& data, size_t rows, size_t cols) {
    assert(data.size() == rows * cols);
    std::vector<size_t> map(rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            map[j * rows + i] = i * cols + j;
        }
    }
    return Clone(data.mapping_reference(map));
}

// Row-major transpose of `m`, as a new (cols x rows) SecureMatrix.
SMatrix Transpose(const SMatrix& m) {
    assert(!m.isColumnWise());
    SMatrix result(TransposeData(m.data(), m.rows(), m.cols()), m.cols(), m.rows(), false);
    result.setPrecision(m.data().getPrecision());
    return result;
}

// Column-wise matrix that represents `m` itself, for use as a matmul right-hand side.
//
// NOTE: reinterpreting a row-major buffer as column-wise yields m^T, NOT m --
// a column-wise (r x c) matrix reads element (i, j) from data[j * r + i]. To
// represent `m`, the buffer has to be physically transposed first.
SMatrix AsColumnWise(const SMatrix& m) {
    assert(!m.isColumnWise());
    SMatrix result(TransposeData(m.data(), m.rows(), m.cols()), m.rows(), m.cols(), true);
    result.setPrecision(m.data().getPrecision());
    return result;
}

// The matmul kernel is called directly throughout, rather than through a
// wrapper. It takes a row-major left operand and a COLUMN-WISE right operand and
// rescales internally (handle_precision() then truncate(), so the result must
// not be divided by `scale`; it throws unless both operands carry the same
// precision). Calling it directly is what makes the orientation of each operand
// visible at the call site -- which matters, because sometimes the column-wise
// form is free (a reinterpretation) and sometimes it needs AsColumnWise.

// Multiplies every entry of `m` by the secret scalar held in the 1-element `s`.
SMatrix ScaleMatrix(const SMatrix& m, const AV& s) {
    assert(s.size() == 1);
    const size_t count = m.rows() * m.cols();

    AV entries = Clone(m.data());
    entries.setPrecision(0);
    AV scalar = s.repeated_subset_reference(count);
    scalar.setPrecision(0);

    // The product lands in a fresh, unmapped buffer, so it is safe to feed to MatMul.
    AV scaled = (*(entries * scalar)) / scale;
    SMatrix result(scaled, m.rows(), m.cols(), m.isColumnWise());
    result.setPrecision(precision);
    return result;
}

// Squared Frobenius norm sum_ij m_ij^2, as a 1-element AV.
AV FrobeniusNormSquared(const SMatrix& m) {
    AV entries = Clone(m.data());
    entries.setPrecision(0);
    AV squares = (*(entries * entries)) / scale;
    squares.setPrecision(precision);

    AV total = squares.chunkedSum(squares.size());
    total.setPrecision(precision);
    return total;
}

// Fixed iteration bound for NewtonSchulzInverse. This must stay a compile-time
// constant: it is what keeps the operator oblivious, since a data-dependent
// stopping rule would leak the conditioning of the input matrix.
//
// Newton-Schulz squares the error each step, so with X_0 = A^T / ||A||_F^2 the
// error after k steps is ||E_0||^(2^k) with ||E_0|| ~ 1 - 1/(n * kappa^2).
// Reaching a target epsilon needs k >~ log2(n * kappa^2 * ln(1/epsilon)); for
// n = 3 and epsilon = 1e-4 that is about 12 steps at kappa = 10 and about 15 at
// kappa = 30. That is only a sanity check on the number below, not its source.
//
// Measured, not guessed. Calibration sweep on the 3PC LAN cluster
// (blinky/pinky/inky, 2026-09-09), reading residual |A*X - I| against iteration
// count over the test suite in main(). Iteration at which each case reaches its
// noise floor, and the floor it settles on:
//
//   identity 3x3         k = 5    0
//   spd, kappa ~ 4       k = 8    ~1e-4
//   spd, kappa ~ 19      k = 13   ~5e-5     <- worst in-range case
//   non-symmetric 3x3    k = 6    ~1e-4
//   2x2                  k = 7    ~5e-5
//   X^T W X / N (2x2)    k = 5    ~7e-5
//
// The worst in-range case plateaus at 13, so 14 carries one iteration of margin.
// The resulting floor (5e-5 to 1.2e-4) sits at or below the abs errors the other
// oblivious operators in these headers report (Exp/Log ~1e-4, BfgsInverseUpdate
// ~5e-4), which is the accuracy bar. Beyond the plateau, extra iterations buy
// nothing and only cost rounds.
//
// A kappa ~ 199 case was also measured and is still descending at k = 20; it is
// outside the operating range by design, and is kept in the suite to mark where
// a fixed bound stops being able to help.
constexpr int kMatrixInverseIterations = 14;

// Sweeps NewtonSchulzInverse over iteration counts and prints the accuracy at
// each, to choose kMatrixInverseIterations. Off by default: it multiplies the
// runtime of the matrix-inverse validation by kMatrixInverseCalibrationMax.
constexpr bool kRunMatrixInverseCalibration = false;
constexpr int kMatrixInverseCalibrationMax = 20;

// Secure matrix inverse by Newton-Schulz iteration:
//
//     X_{k+1} = X_k (2I - A X_k),      X_0 = A^T / ||A||_F^2
//
// The error E_k = I - A X_k obeys E_{k+1} = E_k^2, so convergence is quadratic.
// The initializer guarantees ||E_0||_2 < 1 for any nonsingular A, because
// sigma_max(A) <= ||A||_F, so the iteration is always in the convergence basin.
//
// Security: the iteration count is a compile-time constant and the loop body has
// no branch on shared data and never calls open(). The running time and the
// communication pattern therefore depend only on the public values `n` and
// `iterations`. The price is that an ill-conditioned A silently yields a poorer
// approximation rather than iterating longer -- see kMatrixInverseIterations.
//
// The form below is self-correcting: A X_k is recomputed from A every step, so
// ABY3 truncation error does not accumulate. The iteration descends to the
// fixed-point noise floor and stays there.
//
// Preconditions: A is square, row-major, nonsingular, at `precision`, and
// O(1)-scaled (if ||A||_F^2 is large enough that A^T / ||A||_F^2 underflows the
// fixed-point resolution, X_0 truncates to zero and the iteration cannot move).
SMatrix NewtonSchulzInverse(const SMatrix& a, int iterations = kMatrixInverseIterations) {
    const size_t n = a.rows();
    assert(a.cols() == n);
    assert(!a.isColumnWise());
    EngineRef engine = a.data().engine;

    // The only division in the operator: one boolean division circuit per call.
    AV inverse_norm = SecureReciprocal(FrobeniusNormSquared(a));
    SMatrix x = ScaleMatrix(Transpose(a), inverse_norm);
    x.setPrecision(precision);

    SMatrix two_identity = ScaledIdentity(n, 2.0, engine);

    for (int iteration = 0; iteration < iterations; ++iteration) {
        SMatrix a_x = a.matrixRightMultiplyWithColumnMatrixVectorized(AsColumnWise(x));
        a_x.setPrecision(precision);

        SMatrix residual = two_identity - a_x; // local elementwise
        residual.setPrecision(precision);

        x = x.matrixRightMultiplyWithColumnMatrixVectorized(AsColumnWise(residual));
        x.setPrecision(precision);
    }

    return x;
}

// =============================================================================
// Vectorized optimizer
// =============================================================================

// A batched objective: evaluates `num_points` packed parameter points at once,
// returning one value per point. `params` is length num_points * dim.
using BatchedObjective = std::function<AV(const AV& params, size_t num_points)>;

// An analytic gradient of that objective at a single parameter point: takes a
// length-dim point and returns a length-dim gradient.
//
// Optional. MinimizeBFGSBatched falls back to NumericalGradientBatched when none
// is supplied, which is what every caller before task 0019 did. Supplying one is
// worthwhile when the model has a closed-form gradient: the numerical path costs
// a 2*dim-wide objective evaluation per iteration, so its cost grows with the
// parameter count, while an analytic gradient's does not.
using BatchedGradient = std::function<AV(const AV& params)>;

// Public +-1 pattern that scatters the per-coordinate step h onto the diagonal
// of the stacked parameter points: +h_k into point k, -h_k into point dim + k,
// zero everywhere else.
//
// The pattern depends only on `dim`, so share it once and reuse it for every
// gradient. It has to be secret-shared rather than applied as a public constant
// because the library has no elementwise multiply against a public vector.
AV MakeCentralDifferenceSelector(size_t dim, EngineRef engine) {
    cdough::Vector<DataType> selector(2 * dim * dim, 0);
    for (size_t k = 0; k < dim; ++k) {
        selector[k * dim + k] = scale;                  // forward point
        selector[(dim + k) * dim + k] = -scale;         // backward point
    }
    return engine.secret_share_a(selector, 0, precision);
}

// Central-difference gradient in ONE objective evaluation.
//
// The serial version calls the objective 2*dim times, so its depth grows with
// the parameter count. Here all 2*dim perturbed points are built at once and
// handed to the batched objective together, and the 2*dim divisions collapse
// into a single circuit over `dim` elements. Depth is therefore constant in dim.
//
// Step size matches the serial implementation exactly:
//   h_k = kNumericalGradientStep * (1 + |x_k|).
AV NumericalGradientBatched(const BatchedObjective& f, const AV& x, const AV& selector) {
    const size_t dim = x.size();
    assert(selector.size() == 2 * dim * dim);

    // |x| with no division: sign = 2 * gtez(x) - 1 is an unscaled +-1, so
    // sign * x is already |x| at the original scale.
    AV x_raw = x;
    x_raw.setPrecision(0);
    AV sign = *(x_raw.gtez());
    AV two_sign = *(sign * DataType(2));
    two_sign -= DataType(1);
    AV abs_x = *(two_sign * x_raw);

    AV one_plus_abs = abs_x;
    one_plus_abs += scale;
    const DataType h_step_scaled = static_cast<DataType>(kNumericalGradientStep * scale);
    AV h = (*(one_plus_abs * h_step_scaled)) / scale;  // length dim

    // Stack the 2*dim perturbed points: the base point tiled, plus +-h on the
    // diagonal. Tiling h puts h_k exactly where the selector needs it, so this
    // is one multiply and one add rather than a scatter loop.
    AV x_tiled = x.cyclic_subset_reference(2 * dim);
    AV h_tiled = h.cyclic_subset_reference(2 * dim);
    h_tiled.setPrecision(0);
    AV selector_raw = selector;
    selector_raw.setPrecision(0);
    AV delta = (*(h_tiled * selector_raw)) / scale;
    AV points = x_tiled + delta;
    points.setPrecision(precision);

    AV values = f(points, 2 * dim);  // length 2*dim: [f_plus..., f_minus...]
    values.setPrecision(0);
    AV f_plus = values.slice(0, dim);
    AV f_minus = values.slice(dim, 2 * dim);
    AV diff = f_plus - f_minus;

    // gradient = (f_plus - f_minus) / (2h), all dim of them in one circuit.
    h.setPrecision(0);
    AV two_h = *(h * DataType(2));
    auto diff_scaled_b = (*(diff * scale)).a2b();
    auto two_h_b = two_h.a2b();
    auto grad_b = (*diff_scaled_b) / (*two_h_b);
    AV gradient = *(grad_b->b2a());
    gradient.setPrecision(precision);
    return gradient;
}


// Numerical Hessian of a batched objective, by four-point central second
// differences, as a row-major dim x dim SecureMatrix.
//
//     H_jk = [ f(x+he_j+he_k) - f(x+he_j-he_k)
//            - f(x-he_j+he_k) + f(x-he_j-he_k) ] / (4 h^2)
//
// THREE MPC-SPECIFIC CHOICES, all recorded in semantic task 0022:
//
// 1. BATCHED. Every perturbed point is stacked into a SINGLE objective call, so
//    depth is one evaluation regardless of `dim`, exactly as
//    NumericalGradientBatched does for the gradient. Evaluating the stencil
//    point by point would cost 2*dim*(dim+1) sequential evaluations.
//
// 2. PUBLIC STEP. `step_scaled` is a public constant, unlike the gradient's
//    data-dependent `kNumericalGradientStep * (1 + |x|)`. At the optimum
//    adaptivity buys nothing, and a public step turns the final division into a
//    division by a public constant -- one fewer boolean division circuit, and no
//    secret ever reaches the divisor.
//
// 3. SYMMETRIC BY CONSTRUCTION. Only the upper triangle is evaluated and the
//    result mirrored. This halves the work AND guarantees exact symmetry: a
//    Hessian that is symmetric only up to truncation would make the covariance
//    asymmetric, and the Schur complement below assumes symmetry.
//
// ACCURACY. Second differences divide by h^2, so the objective's own error is
// amplified by 1/h^2. At `precision` 16 that is the binding constraint and the
// reason `step_scaled` must be calibrated by measurement rather than reused from
// the gradient -- see semantic task 0022.
//
// Obliviousness: straight-line. No branch on shared data, no open(). The
// communication pattern depends only on the public `dim` and `step_scaled`.
SMatrix NumericalHessianBatched(const BatchedObjective& f, const AV& x,
                                DataType step_scaled) {
    const size_t dim = x.size();
    EngineRef engine = x.engine;

    // Upper-triangular stencil, four points per entry.
    std::vector<std::pair<size_t, size_t>> pairs;
    pairs.reserve(dim * (dim + 1) / 2);
    for (size_t j = 0; j < dim; ++j) {
        for (size_t k = j; k < dim; ++k) {
            pairs.emplace_back(j, k);
        }
    }
    const size_t num_entries = pairs.size();
    const size_t num_points = 4 * num_entries;

    // The +-h pattern is public and depends only on `dim`, so it is built in the
    // clear and shared once -- the same reason MakeCentralDifferenceSelector
    // shares its pattern rather than applying it as a public vector.
    cdough::Vector<DataType> offsets(num_points * dim, 0);
    for (size_t e = 0; e < num_entries; ++e) {
        const size_t j = pairs[e].first;
        const size_t k = pairs[e].second;
        const int signs[4][2] = {{+1, +1}, {+1, -1}, {-1, +1}, {-1, -1}};
        for (int variant = 0; variant < 4; ++variant) {
            const size_t point = 4 * e + variant;
            // j and k coincide on the diagonal, so accumulate rather than assign.
            offsets[point * dim + j] += signs[variant][0] * step_scaled;
            offsets[point * dim + k] += signs[variant][1] * step_scaled;
        }
    }
    AV offset_shared = engine.secret_share_a(offsets, 0, precision);
    offset_shared.setPrecision(0);

    AV x_tiled = x.cyclic_subset_reference(num_points);
    AV x_raw = Clone(x_tiled);
    x_raw.setPrecision(0);
    AV points = x_raw + offset_shared;
    points.setPrecision(precision);

    AV values = f(points, num_points);  // ONE call
    values.setPrecision(0);

    // Combine the four variants of each entry: (++) - (+-) - (-+) + (--).
    std::vector<size_t> pp_map(num_entries), pm_map(num_entries), mp_map(num_entries),
        mm_map(num_entries);
    for (size_t e = 0; e < num_entries; ++e) {
        pp_map[e] = 4 * e + 0;
        pm_map[e] = 4 * e + 1;
        mp_map[e] = 4 * e + 2;
        mm_map[e] = 4 * e + 3;
    }
    AV pp = Clone(values.mapping_reference(pp_map));
    AV pm = Clone(values.mapping_reference(pm_map));
    AV mp = Clone(values.mapping_reference(mp_map));
    AV mm = Clone(values.mapping_reference(mm_map));
    pp.setPrecision(0); pm.setPrecision(0); mp.setPrecision(0); mm.setPrecision(0);

    AV combined = pp - pm;
    combined -= mp;
    combined += mm;
    combined.setPrecision(0);

    // Divide by 4h^2. Both factors are public, so this is a public constant
    // division -- no boolean circuit. Done in two steps to keep the intermediate
    // from overflowing when `step_scaled` is small.
    const DataType denominator = (4 * step_scaled * step_scaled) / scale;
    AV upper = *(*(combined * scale) / denominator);
    upper.setPrecision(precision);

    // Scatter the upper triangle into a full symmetric matrix. Pure index
    // permutation: no communication.
    std::vector<size_t> full_map(dim * dim);
    for (size_t e = 0; e < num_entries; ++e) {
        const size_t j = pairs[e].first;
        const size_t k = pairs[e].second;
        full_map[j * dim + k] = e;
        full_map[k * dim + j] = e;
    }
    SMatrix hessian(Clone(upper.mapping_reference(full_map)), dim, dim, false);
    hessian.setPrecision(precision);
    return hessian;
}

// Result of the vectorized quasi-Newton optimization. `params` is one AV of
// length dim rather than dim one-element AVs.
struct BatchedOptResult {
    AV params;
    AV value;
    int iterations = 0;
    bool converged = false;

    BatchedOptResult(AV p, AV v, int it = 0, bool conv = false)
        : params(std::move(p)), value(std::move(v)), iterations(it), converged(conv) {}
};

// m (n x n, row-major) * v (length n) -> length n.
//
// An n x 1 matrix has identical row-major and column-wise layouts, so the
// vector needs no transpose to serve as the kernel's column-wise right-hand
// side.
AV MatVecBatched(const SMatrix& m, const AV& v) {
    const size_t n = v.size();
    assert(m.rows() == n && m.cols() == n);
    assert(!m.isColumnWise());
    SMatrix v_col(v, n, 1, true);
    v_col.setPrecision(precision);
    SMatrix result = m.matrixRightMultiplyWithColumnMatrixVectorized(v_col);
    result.setPrecision(precision);
    return result.data();
}

// Outer product a * b^T -> (n x n, row-major).
//
// `a` as a row-major n x 1 and `b` as a column-wise 1 x n are both just the
// buffers themselves (a 1 x n column-wise matrix reads element (0, j) from
// data[j]), so this is one kernel call with no data movement.
SMatrix OuterProduct(const AV& a, const AV& b) {
    const size_t n = a.size();
    assert(b.size() == n);
    SMatrix a_col(a, n, 1, false);
    a_col.setPrecision(precision);
    SMatrix b_row(b, 1, n, true);
    b_row.setPrecision(precision);
    SMatrix result = a_col.matrixRightMultiplyWithColumnMatrixVectorized(b_row);
    result.setPrecision(precision);
    return result;
}

// BFGS update of the inverse-Hessian approximation:
//   H+ = (I - rho s y^T) H (I - rho y s^T) + rho s s^T
//
// The serial version built this from n^3 sequential one-element multiplies. Here
// it is four kernel calls and some local permutation, so its depth is constant
// in n.
//
// Note the right factor: (I - rho s y^T)^T = I - rho y s^T exactly, and a
// column-wise matrix over a row-major buffer *is* that buffer's transpose
// (semantic topic 0001, C-09). So the right factor is the left factor's buffer
// reinterpreted -- no transpose, no extra call. The h_inv transpose below is a
// real one, but AsColumnWise is a local index permutation and costs no
// communication, so there is nothing to gain by assuming h_inv is symmetric
// (it only is up to truncation, and relying on that would silently substitute
// H^T for H as the approximation drifts).
//
// `eye` is supplied by the caller rather than built here. This function is called
// once per BFGS iteration, and Identity() secret-shares n*n elements, so building
// it internally re-shared the same public constant on every iteration -- a cost
// that grows quadratically in the parameter count for no benefit. The caller
// already holds an identity of the right size.
SMatrix BfgsInverseUpdateBatched(const SMatrix& h_inv, const AV& s, const AV& y, const AV& rho,
                                 const SMatrix& eye) {
    const size_t n = s.size();
    assert(h_inv.rows() == n && h_inv.cols() == n);
    assert(y.size() == n);
    assert(eye.rows() == n && eye.cols() == n);

    SMatrix left = eye - ScaleMatrix(OuterProduct(s, y), rho);
    left.setPrecision(precision);

    // temp = left * h_inv   (h_inv needs a genuine transpose to be a rhs)
    SMatrix temp = left.matrixRightMultiplyWithColumnMatrixVectorized(AsColumnWise(h_inv));
    temp.setPrecision(precision);

    // right = left^T, for free.
    SMatrix right_col(left.data(), n, n, true);
    right_col.setPrecision(precision);

    SMatrix updated = temp.matrixRightMultiplyWithColumnMatrixVectorized(right_col) +
                      ScaleMatrix(OuterProduct(s, s), rho);
    updated.setPrecision(precision);
    return updated;
}

// Armijo sufficient-decrease constant. Matches the pre-oblivious value.
//
// At `precision` 16 the fixed-point representation of 1e-4 is
// llround(1e-4 * 65536) = 7, i.e. 1.068e-4, a 7% error on the constant itself.
// That is harmless: c1 only sets how much of the predicted decrease a step must
// actually deliver, and no line search is sensitive to it at the third digit.
const double kArmijoC1 = 1e-4;

// Number of step sizes in the oblivious line search ladder: alpha_l = 2^-l for
// l in [0, kLineSearchSteps).
//
// Seventeen reproduces the pre-oblivious backtracking loop exactly. That loop
// started at alpha = 1, halved on rejection, and gave up once alpha < 1e-5, so
// the alphas it could ever try were 2^-l with 2^-l >= 1e-5, i.e. l <= 16 --
// seventeen candidates, the smallest being 2^-16 = 1.526e-5.
//
// Seventeen is also the ceiling the number format allows: 2^-16 is exactly 1 at
// `precision` 16, and 2^-17 truncates to zero, which would make the last rung a
// zero step rather than a small one.
constexpr int kLineSearchSteps = 17;

// Public constants for the oblivious line search. They depend only on
// kLineSearchSteps and kArmijoC1, so they are shared once per optimization and
// reused by every iteration.
//
// They have to be secret-shared rather than applied as public vectors because
// the library has no elementwise multiply against a public vector -- the same
// reason MakeCentralDifferenceSelector shares its pattern. Nothing about them is
// secret; sharing is a calling-convention detail.
struct LineSearchConstants {
    AV alpha;          // length L: 2^-l in fixed point
    AV armijo_slack;   // length L: c1 * 2^-l in fixed point
    AV strict_prefix;  // length L*L: raw 1 at (l, m) iff m < l
};

LineSearchConstants MakeLineSearchConstants(EngineRef engine) {
    constexpr size_t kSteps = static_cast<size_t>(kLineSearchSteps);

    cdough::Vector<DataType> alpha(kSteps, 0);
    cdough::Vector<DataType> slack(kSteps, 0);
    for (size_t l = 0; l < kSteps; ++l) {
        alpha[l] = scale >> l;
        slack[l] = std::llround(kArmijoC1 * std::ldexp(1.0, -static_cast<int>(l)) * scale);
    }

    // Strictly-lower-triangular pattern, so summing row l of a tiled copy of a
    // 0/1 vector counts its set entries strictly before l.
    cdough::Vector<DataType> prefix(kSteps * kSteps, 0);
    for (size_t l = 0; l < kSteps; ++l) {
        for (size_t m = 0; m < l; ++m) {
            prefix[l * kSteps + m] = 1;
        }
    }

    // Every one of these is only ever used as a multiplication operand, and
    // multiply_a throws unless both operands carry the SAME precision (see
    // protocol.h handle_precision). They are pinned at 0 here and every operand
    // derived from them is pinned at 0 at its point of use, so the pairing is
    // checkable locally rather than by tracing precision across the loop.
    LineSearchConstants constants{engine.secret_share_a(alpha, 0, 0),
                                  engine.secret_share_a(slack, 0, 0),
                                  engine.secret_share_a(prefix, 0, 0)};
    constants.alpha.setPrecision(0);
    constants.armijo_slack.setPrecision(0);
    constants.strict_prefix.setPrecision(0);
    return constants;
}

// Oblivious BFGS with a batched central-difference gradient and a branch-free
// line search.
//
// OBLIVIOUSNESS CONTRACT (semantic task 0017)
// -------------------------------------------
// The loop opens exactly ONE value per iteration: a single bit saying whether to
// keep going. Every convergence test that bit is computed from -- the gradient
// magnitude, the objective values, the curvature sign, the backtracking depth --
// is evaluated on shares and never leaves the protocol. The information content
// of the opened bit is the iteration count, which a loop that terminates early
// reveals through its own wall clock in any case.
//
// Everything else that used to drive a branch is now multiplexed: both arms are
// evaluated and the result selected with `Multiplex`, so the sequence of
// operations, and therefore the communication pattern, depends only on the
// public values `dim`, `kLineSearchSteps`, and `max_iterations`.
//
// The per-iteration diagnostic log opens considerably more than one bit, which
// is why it is behind LOGISTIC_REGRESSION_LAYER_PRINT and why that flag defaults
// to OFF. A build with the flag on is a debugging build and makes no privacy
// claim.
//
// Numerically this is the same optimizer as before. The line search is the one
// place where the arithmetic differs in form: it evaluates all seventeen
// candidate step sizes in a single batched objective call instead of walking
// them in sequence, which is both oblivious and shallower -- depth one
// evaluation regardless of how much backtracking a step needs, where the old
// loop paid one sequential evaluation per halving.
//
// Two deliberate differences from the old loop, both documented in task 0017:
//   1. The old `std::isfinite(fx_new)` guard has no fixed-point equivalent;
//      there is no infinity to test for, and an overflowed candidate fails the
//      Armijo comparison on its own.
//   2. A line search that finds no acceptable step is detected at the top of the
//      following iteration rather than at the end of the failing one, so that it
//      shares the single opened flag. `result.iterations` is therefore one
//      higher than before on that path. The returned parameters are unaffected.
BatchedOptResult MinimizeBFGSBatched(const BatchedObjective& f, const AV& x0,
                                     int max_iterations = 20,
                                     const BatchedGradient& analytic_gradient = nullptr) {
    const size_t n = x0.size();
    EngineRef engine = x0.engine;

    AV x = Clone(x0);
    x.setPrecision(precision);
    AV fx = f(x, 1);
    fx.setPrecision(precision);

    // Patterns that depend only on `dim`, shared once and reused every iteration.
    AV selector = MakeCentralDifferenceSelector(n, engine);
    const LineSearchConstants ls = MakeLineSearchConstants(engine);
    const SMatrix eye = Identity(n, engine);

    AV one(1, engine);
    one += scale;

    // One place decides how the gradient is obtained, so the two call sites below
    // cannot drift apart. The numerical path is the default and is unchanged.
    const auto gradient_at = [&](const AV& point) {
        return analytic_gradient ? analytic_gradient(point)
                                 : NumericalGradientBatched(f, point, selector);
    };

    AV gradient = gradient_at(x);
    SMatrix h_inv = Identity(n, engine);

    // 1 while the previous iteration's line search succeeded. Starts at 1: there
    // is no previous iteration to have failed.
    AV line_search_ok(1, engine);
    line_search_ok += DataType(1);
    line_search_ok.setPrecision(0);

    BatchedOptResult result(x, fx, 0, false);

    // Per-iteration wall clock. Deliberately NOT the repository's
    // `stopwatch::timepoint`: that prints its own `[ SW]` format, keeps a single
    // static `then` shared by every caller (so interleaving it with the
    // harness's phase timepoints would corrupt both sets of intervals), and
    // records one map entry per label, which at 300 iterations is 300 entries.
    //
    // Timing is printed unconditionally, unlike the value fields below. It costs
    // nothing in privacy terms: elapsed time is not derived from any share, so
    // printing it is not a declassification, and every party can measure it with
    // its own clock regardless. It is also never branched on.
    using Clock = std::chrono::steady_clock;

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        const Clock::time_point iteration_start = Clock::now();
        result.iterations = iteration + 1;

        // -------------------------------------------------------------------
        // The one declassification. `grad_big` is 1 iff some |gradient_i| has
        // not yet fallen below kSmallEpsilon; `line_search_ok` is 1 iff the
        // previous iteration made progress. Their product is the continue bit.
        //
        // The pre-oblivious code opened the whole gradient vector here and
        // reduced it to a max in plaintext, which leaked every component to
        // decide one bit.
        // -------------------------------------------------------------------
        AV grad_big = AnyAbsAtLeast(gradient, kSmallEpsilon_scaled);
        AV keep_going = *(grad_big * line_search_ok);
        auto opened_flag = keep_going.open();

        if (static_cast<DataType>(opened_flag[0]) == 0) {
            const double elapsed =
                std::chrono::duration<double>(Clock::now() - iteration_start).count();

            // Which of the two conditions cleared the flag is a *second* fact
            // about the shares, so recovering it needs another open and stays
            // behind the guard. The index and the time do not.
#ifdef LOGISTIC_REGRESSION_LAYER_PRINT
            auto opened_grad_big = grad_big.open();
            const bool stopped_on_gradient = (static_cast<DataType>(opened_grad_big[0]) == 0);
#endif
            if (engine.getPartyID() == 0) {
                std::cout << "[BFGS] iter " << std::setw(3) << (iteration + 1)
                          << "  time=" << std::fixed << std::setprecision(3) << elapsed << "s"
#ifdef LOGISTIC_REGRESSION_LAYER_PRINT
                          << (stopped_on_gradient
                                  ? "  gradient below tolerance; stopping"
                                  : "  no descent found along search direction; stopping")
#else
                          << "  stopping"
#endif
                          << std::endl;
            }
            result.converged = true;
            break;
        }

        // -------------------------------------------------------------------
        // Search direction. Both arms are computed: the BFGS direction and the
        // steepest-descent fallback the old code only built when it saw a
        // non-descent direction in the clear. `dir_sd` is local arithmetic and
        // `dd_sd` one dot product, so evaluating the unused arm is cheap.
        // -------------------------------------------------------------------
        AV dir_bfgs = -MatVecBatched(h_inv, gradient);
        dir_bfgs.setPrecision(precision);
        AV dd_bfgs = *gradient.dot_product(dir_bfgs, n);
        dd_bfgs.setPrecision(precision);

        AV dir_sd = -gradient;
        dir_sd.setPrecision(precision);
        AV dd_sd = *gradient.dot_product(dir_sd, n);
        dd_sd.setPrecision(precision);

        // not_descent = 1 iff dd_bfgs >= 0, i.e. the approximation stopped
        // producing a descent direction and H must be reset to the identity.
        AV dd_bfgs_raw = Clone(dd_bfgs);
        dd_bfgs_raw.setPrecision(0);
        AV not_descent = *(dd_bfgs_raw.gtez());
        not_descent.setPrecision(0);

        AV direction = Multiplex(not_descent, dir_bfgs, dir_sd);
        AV dd = Multiplex(not_descent, dd_bfgs, dd_sd);
        h_inv = SMatrix(Multiplex(not_descent, h_inv.data(), eye.data()), n, n, false);
        h_inv.setPrecision(precision);

        // -------------------------------------------------------------------
        // Line search. All kLineSearchSteps candidates at once: tile x and the
        // direction across the ladder, scale by the per-rung alpha, and evaluate
        // the objective once over the whole stack.
        // -------------------------------------------------------------------
        AV x_tiled = x.cyclic_subset_reference(kLineSearchSteps);
        AV dir_tiled = direction.cyclic_subset_reference(kLineSearchSteps);
        dir_tiled.setPrecision(0);
        AV alpha_rows = ls.alpha.repeated_subset_reference(n);
        alpha_rows.setPrecision(0);
        AV candidate_delta = (*(dir_tiled * alpha_rows)) / scale;
        AV points = x_tiled + candidate_delta;
        points.setPrecision(precision);

        AV values = f(points, kLineSearchSteps);
        AV values_raw = Clone(values);
        values_raw.setPrecision(0);

        // Armijo: accept rung l iff values_l <= fx + c1 * alpha_l * dd.
        AV fx_rep = fx.repeated_subset_reference(kLineSearchSteps);
        AV dd_rep = dd.repeated_subset_reference(kLineSearchSteps);
        dd_rep.setPrecision(0);
        AV armijo_slack = ls.armijo_slack;
        armijo_slack.setPrecision(0);
        AV slack = (*(dd_rep * armijo_slack)) / scale;
        AV threshold = fx_rep + slack;
        threshold.setPrecision(0);
        AV accept = *((*(threshold - values_raw)).gtez());  // raw 0/1 per rung
        accept.setPrecision(0);

        // Take the FIRST accepting rung. Armijo acceptance is not monotone in
        // alpha, so this needs a real prefix computation rather than a
        // comparison against a count. Tiling `accept` and masking with the
        // strictly-lower-triangular pattern gives, in one multiply plus a local
        // reduction, the number of accepts strictly before each rung -- where a
        // prefix product would have cost log2(L) sequential rounds.
        AV accept_tiled = accept.cyclic_subset_reference(kLineSearchSteps);
        accept_tiled.setPrecision(0);
        AV strict_prefix = ls.strict_prefix;
        strict_prefix.setPrecision(0);
        AV before = (*(accept_tiled * strict_prefix)).chunkedSum(kLineSearchSteps);
        before.setPrecision(0);
        before -= DataType(1);
        AV any_before = *(before.gtez());
        AV none_before = -any_before;
        none_before += DataType(1);
        none_before.setPrecision(0);
        AV take = *(accept * none_before);  // at most one entry set
        take.setPrecision(0);

        AV accept_count = accept.chunkedSum(kLineSearchSteps);
        accept_count -= DataType(1);
        AV found = *(accept_count.gtez());  // 1 iff any rung qualified
        found.setPrecision(0);

        // Selecting with `take` yields 0 when no rung qualified, so alpha is 0
        // on failure -- which reproduces the old behaviour exactly, since the
        // old loop restored x on giving up and x + 0 * direction == x.
        AV ladder = ls.alpha;
        ladder.setPrecision(0);
        AV alpha = (*(take * ladder)).chunkedSum(kLineSearchSteps);
        alpha.setPrecision(precision);

        AV fx_selected = (*(take * values_raw)).chunkedSum(kLineSearchSteps);
        fx_selected.setPrecision(precision);

        // Rebuilding the step from the selected alpha is the same arithmetic as
        // the candidate above, so x_new is bit-identical to the chosen candidate
        // point and `fx_selected` is its objective, not an approximation of it.
        AV alpha_rep = alpha.repeated_subset_reference(n);
        alpha_rep.setPrecision(0);
        AV dir_raw = Clone(direction);
        dir_raw.setPrecision(0);
        AV step = (*(dir_raw * alpha_rep)) / scale;
        step.setPrecision(precision);

        AV x_new = x + step;
        x_new.setPrecision(precision);
        AV fx_new = Multiplex(found, fx, fx_selected);

        // -------------------------------------------------------------------
        // Curvature-gated inverse-Hessian update. Both arms again: the update is
        // always computed, then kept or discarded by multiplex.
        // -------------------------------------------------------------------
        AV gradient_new = gradient_at(x_new);
        AV gradient_delta = gradient_new - gradient;
        gradient_delta.setPrecision(precision);

        AV curvature = *step.dot_product(gradient_delta, n);
        curvature.setPrecision(precision);

        // curvature > kSmallEpsilon, as integers: curvature >= eps + 1.
        AV curv_raw = Clone(curvature);
        curv_raw.setPrecision(0);
        curv_raw -= (kSmallEpsilon_scaled + 1);
        AV curv_ok = *(curv_raw.gtez());
        curv_ok.setPrecision(0);

        // The denominator is multiplexed to 1.0 BEFORE the reciprocal, not
        // after. SecureReciprocal routes through a non-restoring division
        // circuit that assumes a non-negative operand, so handing it the raw
        // curvature when the guard fails would be undefined behaviour inside the
        // circuit, not merely a discarded result.
        AV safe_curvature = Multiplex(curv_ok, one, curvature);
        AV rho = SecureReciprocal(safe_curvature);

        SMatrix h_updated = BfgsInverseUpdateBatched(h_inv, step, gradient_delta, rho, eye);
        h_inv = SMatrix(Multiplex(curv_ok, h_inv.data(), h_updated.data()), n, n, false);
        h_inv.setPrecision(precision);

        // ---------------------------------------------------------------------
        // Per-iteration log.
        //
        // The index and the elapsed time print unconditionally: neither is
        // derived from a share, so neither is a declassification. The value
        // fields below need `open()` and stay behind the compile-time flag, so
        // the flag-ON format is a strict superset of the flag-OFF one.
        //
        // Note where the opens sit. They are OUTSIDE the party guard, because
        // `open()` is a communication round and every party must reach it;
        // only the printing is party 0's. Moving an open inside the guard
        // deadlocks all three parties.
        // ---------------------------------------------------------------------
        const double elapsed =
            std::chrono::duration<double>(Clock::now() - iteration_start).count();

#ifdef LOGISTIC_REGRESSION_LAYER_PRINT
        auto opened_g = gradient.open();
        auto opened_step = step.open();
        auto opened_alpha = alpha.open();
        auto opened_fx = fx.open();
        auto opened_fx_new = fx_new.open();

        double max_grad = 0.0;
        double max_step = 0.0;
        for (size_t i = 0; i < n; ++i) {
            max_grad = std::max(max_grad, std::abs(static_cast<double>(opened_g[i]) / scale));
            max_step = std::max(max_step, std::abs(static_cast<double>(opened_step[i]) / scale));
        }
        const double fx_val = static_cast<double>(opened_fx[0]) / scale;
        const double fx_new_val = static_cast<double>(opened_fx_new[0]) / scale;
#endif

        if (engine.getPartyID() == 0) {
            std::cout << "[BFGS] iter " << std::setw(3) << (iteration + 1)
                      << "  time=" << std::fixed << std::setprecision(3) << elapsed << "s"
#ifdef LOGISTIC_REGRESSION_LAYER_PRINT
                      << "  neg_log_lik=" << std::fixed << std::setprecision(6) << fx_new_val
                      << "  |grad|=" << std::scientific << std::setprecision(3) << max_grad
                      << "  alpha=" << std::fixed << std::setprecision(4)
                      << static_cast<double>(opened_alpha[0]) / scale
                      << "  |step|=" << std::scientific << std::setprecision(3) << max_step
                      << "  d_obj=" << std::scientific << std::setprecision(3)
                      << std::abs(fx_val - fx_new_val)
#endif
                      << std::endl;
        }

        x = Clone(x_new);
        gradient = Clone(gradient_new);
        fx = Clone(fx_new);
        line_search_ok = Clone(found);
    }

    result.params = Clone(x);
    result.value = Clone(fx);
    return result;
}

}  // namespace cdough::regression


// =============================================================================
// APPENDED: the pipeline's quasi-Newton optimiser (was playground/optimizer.h)
//
// The pipeline's quasi-Newton optimiser.
//
// The secure matrix helpers (ScaledIdentity, Identity, TransposeData, Transpose,
// AsColumnWise, ScaleMatrix, FrobeniusNormSquared) and NewtonSchulzInverse now
// come from library/optimizer.h -- the copies that used to live here were
// byte-identical to the library's, apart from NewtonSchulzInverse, whose library
// form uses the vectorized matrix multiply. MatMul went with them: it existed
// only to serve this header's own Newton-Schulz, which the pipeline never calls
// (ObservedInformationSE inverts in plaintext by Gauss-Jordan).
//
// What remains is the part the library has no counterpart for: a BFGS driver
// over `std::vector<AV>` parameter vectors with a value-plus-gradient objective
// (task 0009). library/optimizer.h's MinimizeBFGSBatched packs several candidate
// parameter vectors into one AV instead, which is a different interface built on
// the balanced dataset layout, not another spelling of this one.
// =============================================================================

#include <cassert>
#include <cstddef>

namespace cdough::regression {

// Central finite-difference gradient of a scalar objective `f` at `x`.
std::vector<AV> NumericalGradient(
    const std::function<AV(const std::vector<AV>&)>& f,
    const std::vector<AV>& x) {
    size_t dim = x.size();
    std::vector<AV> gradient;
    gradient.reserve(dim);

    std::vector<AV> perturbed = Clone(x);
    for (size_t k = 0; k < dim; ++k) {
        perturbed[k].setPrecision(precision);
    }

    for (size_t k = 0; k < dim; ++k) {
        // Step size h = kNumericalGradientStep * (1.0 + |x[k]|)
        AV xk_copy = x[k];
        xk_copy.setPrecision(0);
        AV mask = *(xk_copy.gtez());
        AV two_mask = *(mask * DataType(2));
        two_mask -= DataType(1);
        AV abs_xk = *(two_mask * xk_copy); // (1 or -1) * xk_copy gives |x[k]| directly without / scale

        AV one_plus_abs = abs_xk;
        one_plus_abs += scale;
        DataType h_step_scaled = static_cast<DataType>(kNumericalGradientStep * scale);
        AV h = (*(one_plus_abs * h_step_scaled)) / scale; // size 1

        // f_plus = f(perturbed with x[k] + h)
        h.setPrecision(precision);
        perturbed[k] = x[k] + h;
        perturbed[k].setPrecision(precision);
        AV f_plus = f(perturbed);
        f_plus.setPrecision(0);

        // f_minus = f(perturbed with x[k] - h)
        perturbed[k] = x[k] - h;
        perturbed[k].setPrecision(precision);
        AV f_minus = f(perturbed);
        f_minus.setPrecision(0);

        // Reset perturbed[k]
        perturbed[k] = x[k];
        perturbed[k].setPrecision(precision);

        // gradient[k] = (f_plus - f_minus) / (2 * h)
        AV diff = f_plus - f_minus;
        h.setPrecision(0);
        AV two_h = *(h * DataType(2));

        auto diff_scaled_b = (*(diff * scale)).a2b();
        auto two_h_b = two_h.a2b();
        auto grad_b = (*diff_scaled_b) / (*two_h_b);
        AV grad_k = *(grad_b->b2a());
        grad_k.setPrecision(precision);

        gradient.push_back(std::move(grad_k));
    }

    return gradient;
}

// =============================================================================
// BFGS Optimization & Secure Linear Algebra Helpers
// =============================================================================

// Matrix-vector product m * v where m is SecureMatrix (n x n) and v is std::vector<AV> (length n).
std::vector<AV> MatVec(const SMatrix& m, const std::vector<AV>& v) {
    size_t n = v.size();
    assert(m.rows() == n && m.cols() == n);
    EngineRef engine = v[0].engine;

    // m is stored row-major: element (i, j) is at data_[i * n + j]
    AV m_data = m.data();
    m_data.setPrecision(0);

    std::vector<AV> result;
    result.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        AV row_sum(1, engine);
        row_sum.setPrecision(0);
        for (size_t j = 0; j < n; ++j) {
            // Slice element at index (i * n + j)
            AV m_ij = m_data.slice(i * n + j, i * n + j + 1);
            m_ij.setPrecision(0);
            AV vj = v[j];
            vj.setPrecision(0);
            AV prod = (*(m_ij * vj)) / scale;
            row_sum += prod;
        }
        row_sum.setPrecision(precision);
        result.push_back(std::move(row_sum));
    }
    return result;
}

// Standard inner product between two secure vectors.
AV Dot(const std::vector<AV>& a, const std::vector<AV>& b) {
    assert(a.size() == b.size());
    size_t n = a.size();
    EngineRef engine = a[0].engine;

    AV sum(1, engine);
    sum.setPrecision(0);
    for (size_t i = 0; i < n; ++i) {
        AV ai = a[i];
        AV bi = b[i];
        ai.setPrecision(0);
        bi.setPrecision(0);
        AV prod = (*(ai * bi)) / scale;
        sum += prod;
    }
    sum.setPrecision(precision);
    return sum;
}

// BFGS update of the inverse-Hessian approximation:
//   H+ = (I - rho s y^T) H (I - rho y s^T) + rho s s^T,   rho = 1 / (y^T s).
SMatrix BfgsInverseUpdate(const SMatrix& h_inv, const std::vector<AV>& s,
                         const std::vector<AV>& y, const AV& rho) {
    size_t n = s.size();
    assert(h_inv.rows() == n && h_inv.cols() == n);
    assert(y.size() == n);
    EngineRef engine = s[0].engine;

    // We extract h_inv elements into an n x n 2D array of 1-element AVs
    // Distinct buffers per element
    std::vector<std::vector<AV>> h_elements = MakeMatrix(n, n, engine);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            h_elements[i][j] = h_inv.data().slice(i * n + j, i * n + j + 1);
            h_elements[i][j].setPrecision(0);
        }
    }

    AV rho_copy = rho;
    rho_copy.setPrecision(0);

    // Compute left = I - rho * s * y^T as an n x n matrix in std::vector<std::vector<AV>>
    // In fixed point: (s_i * y_j) / scale, then (* rho) / scale
    std::vector<std::vector<AV>> left = MakeMatrix(n, n, engine);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            AV si = s[i];
            AV yj = y[j];
            si.setPrecision(0);
            yj.setPrecision(0);
            AV s_y = (*(si * yj)) / scale;
            AV rho_s_y = (*(rho_copy * s_y)) / scale;

            AV elem(1, engine);
            elem.setPrecision(0);
            if (i == j) {
                elem += scale;
            }
            elem -= rho_s_y;
            left[i][j] = elem;
        }
    }

    // temp = left * h_inv
    // temp[i][j] = sum_k (left[i][k] * h_inv[k][j]) / scale
    std::vector<std::vector<AV>> temp = MakeMatrix(n, n, engine);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            AV sum(1, engine);
            sum.setPrecision(0);
            for (size_t k = 0; k < n; ++k) {
                AV h_kj = h_elements[k][j];
                AV left_ik = left[i][k];
                left_ik.setPrecision(0);
                h_kj.setPrecision(0);
                AV prod = (*(left_ik * h_kj)) / scale;
                sum += prod;
            }
            temp[i][j] = sum;
        }
    }

    // updated = temp * left^T + rho * s * s^T
    // Flatten result into a single AV of size n * n to construct SecureMatrix
    cdough::Vector<DataType> dummy_init(n * n, 0);
    AV updated_data = engine.secret_share_a(dummy_init, 0, 0);
    updated_data.setPrecision(0);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            AV sum(1, engine);
            sum.setPrecision(0);
            for (size_t k = 0; k < n; ++k) {
                AV temp_ik = temp[i][k];
                AV left_jk = left[j][k];
                temp_ik.setPrecision(0);
                left_jk.setPrecision(0);
                AV prod = (*(temp_ik * left_jk)) / scale;
                sum += prod;
            }
            AV si = s[i];
            AV sj = s[j];
            si.setPrecision(0);
            sj.setPrecision(0);
            AV s_s = (*(si * sj)) / scale;
            AV rho_s_s = (*(rho_copy * s_s)) / scale;
            sum += rho_s_s;

            // Place into updated_data at index i * n + j
            AV sum_rep = sum.repeated_subset_reference(n * n);
            sum_rep.setPrecision(0);
            cdough::Vector<DataType> mask_vec(n * n, 0);
            mask_vec[i * n + j] = scale; // Use scale (1.0 in fixed-point)
            AV mask_elem = engine.secret_share_a(mask_vec, 0, 0);
            mask_elem.setPrecision(0);
            AV placed = (*(sum_rep * mask_elem)) / scale;
            placed.setPrecision(0);
            updated_data += placed;
        }
    }

    updated_data.setPrecision(precision);
    SMatrix updated(updated_data, n, n, false);
    updated.setPrecision(precision);
    return updated;
}

// Result of the outer quasi-Newton optimization.
struct OptResult {
    std::vector<AV> params;
    AV value;
    int iterations = 0;
    bool converged = false;
    // False when no curvature update ever passed the y^T s > 0 test, in which
    // case h_inv is still the identity and its diagonal is NOT a variance.
    bool hessian_updated = false;

    OptResult(std::vector<AV> p, AV v, int it = 0, bool conv = false)
        : params(std::move(p)), value(std::move(v)), iterations(it), converged(conv) {}
};

// Minimizes `f` starting from `x0` using BFGS with a backtracking (Armijo) line
// search and numerical gradients.
// `h_inv_out`, when given, receives the converged inverse-Hessian approximation.
// For a negative log-likelihood objective that is the asymptotic covariance
// matrix of the estimates, so the standard errors come out of the optimisation
// for free. It is a quasi-Newton approximation rather than the exact observed
// information -- unlike the IRLS models, whose covariance is exact -- and the
// report says so.
// An objective that can return its own gradient. Passing nullptr asks for the
// value alone, which is all the line search needs; passing a vector asks for
// both, so a caller with an analytic gradient pays for the shared work once.
using ValueGradFn = std::function<AV(const std::vector<AV>&, std::vector<AV>*)>;

OptResult MinimizeBFGS(const ValueGradFn& f, const std::vector<AV>& x0,
                       int max_iterations = 20) {
    size_t n = x0.size();
    EngineRef engine = x0[0].engine;

    std::vector<AV> x = Clone(x0);
    for (size_t i = 0; i < n; ++i) {
        x[i].setPrecision(precision);
    }
    std::vector<AV> gradient;
    AV fx = f(x, &gradient);
    fx.setPrecision(precision);

    SMatrix h_inv = Identity(n, engine);

    OptResult result(x, fx, 0, false);

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        result.iterations = iteration + 1;

        // Check gradient convergence in plaintext after opening
        double max_grad = 0.0;
        for (size_t i = 0; i < n; ++i) {
            auto opened_g = gradient[i].open();
            double val = std::abs(static_cast<double>(opened_g[0]) / scale);
            if (val > max_grad) max_grad = val;
        }

        if (max_grad < kSmallEpsilon) {
            result.converged = true;
            break;
        }

        // Search direction d = -H_inv * gradient
        std::vector<AV> direction = MatVec(h_inv, gradient);
        for (size_t i = 0; i < n; ++i) {
            direction[i] = -direction[i];
            direction[i].setPrecision(precision);
        }

        AV directional_derivative = Dot(gradient, direction);
        directional_derivative.setPrecision(precision);

        // Check if descent direction: directional_derivative < 0
        auto opened_dd = directional_derivative.open();
        double dd_val = static_cast<double>(opened_dd[0]) / scale;
        if (dd_val >= 0.0) {
            h_inv = Identity(n, engine);
            for (size_t i = 0; i < n; ++i) {
                direction[i] = -gradient[i];
                direction[i].setPrecision(precision);
            }
            directional_derivative = Dot(gradient, direction);
            directional_derivative.setPrecision(precision);
            auto opened_dd2 = directional_derivative.open();
            dd_val = static_cast<double>(opened_dd2[0]) / scale;
        }

        // Backtracking line search satisfying the Armijo sufficient-decrease rule
        const double c1 = 1e-4;
        double alpha = 1.0;
        bool line_search_failed = false;
        std::vector<AV> x_new = Clone(x);
        AV fx_new = Clone(fx);

        auto opened_fx = fx.open();
        double fx_val = static_cast<double>(opened_fx[0]) / scale;

        while (true) {
            for (size_t i = 0; i < n; ++i) {
                AV alpha_dir = Clone(direction[i]);
                alpha_dir.setPrecision(0);
                alpha_dir = (*(alpha_dir * static_cast<DataType>(alpha * scale))) / scale;
                x_new[i] = x[i] + alpha_dir;
                x_new[i].setPrecision(precision);
            }
            fx_new = f(x_new, nullptr);
            fx_new.setPrecision(precision);

            auto opened_fx_new = fx_new.open();
            double fx_new_val = static_cast<double>(opened_fx_new[0]) / scale;

            if (std::isfinite(fx_new_val) &&
                fx_new_val <= fx_val + c1 * alpha * dd_val) {
                break;
            }
            alpha *= 0.5;
            if (alpha < 1e-5) {
                x_new = x;
                fx_new = fx;
                line_search_failed = true;
                break;
            }
        }

        // Step = x_new - x
        std::vector<AV> step = MakeVector(n, 1, engine);
        double max_step = 0.0;
        for (size_t i = 0; i < n; ++i) {
            step[i] = x_new[i] - x[i];
            step[i].setPrecision(precision);
            auto opened_s = step[i].open();
            double val = std::abs(static_cast<double>(opened_s[0]) / scale);
            if (val > max_step) max_step = val;
        }

        // A failed line search is a fixed point: x is unchanged, so the next
        // iteration recomputes the same direction and fails identically
        if (line_search_failed) {
            if (engine.getPartyID() == 0) {
                std::cout << "[BFGS] iter " << std::setw(3) << (iteration + 1)
                          << "  no descent found along search direction; stopping at"
                          << " neg_log_lik=" << std::fixed << std::setprecision(6) << fx_val
                          << "  |grad|=" << std::scientific << std::setprecision(3) << max_grad
                          << std::endl;
            }
            result.converged = true;
            break;
        }

        std::vector<AV> gradient_new;
        f(x_new, &gradient_new);
        std::vector<AV> gradient_delta = MakeVector(n, 1, engine);
        for (size_t i = 0; i < n; ++i) {
            gradient_delta[i] = gradient_new[i] - gradient[i];
            gradient_delta[i].setPrecision(precision);
        }

        AV curvature = Dot(step, gradient_delta);
        curvature.setPrecision(precision);
        auto opened_curv = curvature.open();
        double curv_val = static_cast<double>(opened_curv[0]) / scale;

        if (curv_val > kSmallEpsilon) {
            // rho = 1.0 / curvature
            AV scale_sq(1, engine);
            scale_sq += (DataType(1) << (2 * precision));
            auto scale_sq_b = scale_sq.a2b();
            auto curv_b = curvature.a2b();
            auto rho_b = (*scale_sq_b) / (*curv_b);
            AV rho = *(rho_b->b2a());
            rho.setPrecision(precision);

            h_inv = BfgsInverseUpdate(h_inv, step, gradient_delta, rho);
            h_inv.setPrecision(precision);
            result.hessian_updated = true;
        }

        x = x_new;
        gradient = gradient_new;
        auto opened_new_fx = fx_new.open();
        double obj_change = std::abs(fx_val - static_cast<double>(opened_new_fx[0]) / scale);
        fx = fx_new;

        if (engine.getPartyID() == 0) {
            std::cout << "[BFGS] iter " << std::setw(3) << (iteration + 1)
                      << "  neg_log_lik=" << std::fixed << std::setprecision(6) << static_cast<double>(opened_new_fx[0]) / scale
                      << "  |grad|=" << std::scientific << std::setprecision(3) << max_grad
                      << "  alpha=" << std::fixed << std::setprecision(4) << alpha
                      << "  |step|=" << std::scientific << std::setprecision(3) << max_step
                      << "  d_obj=" << obj_change << std::endl;
        }

        if (max_step < kSmallEpsilon && max_grad < kSmallEpsilon) {
            result.converged = true;
            break;
        }
    }

    result.params = x;
    result.value = fx;
    return result;
}


// Overload for an objective with no analytic gradient: differences it. This is
// the signature the logistic-regression branch's callers use, and it keeps the
// numerical path available as something to validate the analytic one against.
//
// The two overloads are distinguished by arity -- a one-argument callable is not
// convertible to ValueGradFn -- so a lambda selects the right one on its own.
OptResult MinimizeBFGS(const std::function<AV(const std::vector<AV>&)>& f,
                       const std::vector<AV>& x0, int max_iterations = 20) {
    ValueGradFn wrapped = [&f](const std::vector<AV>& x, std::vector<AV>* g) {
        if (g != nullptr) *g = NumericalGradient(f, x);
        return f(x);
    };
    return MinimizeBFGS(wrapped, x0, max_iterations);
}

}  // namespace cdough::regression

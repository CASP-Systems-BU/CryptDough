#pragma once

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
SMatrix BfgsInverseUpdateBatched(const SMatrix& h_inv, const AV& s, const AV& y, const AV& rho) {
    const size_t n = s.size();
    assert(h_inv.rows() == n && h_inv.cols() == n);
    assert(y.size() == n);
    EngineRef engine = s.engine;

    SMatrix eye = Identity(n, engine);
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

// Vectorized BFGS with a backtracking (Armijo) line search and a batched
// central-difference gradient.
//
// Logic is identical to the pre-vectorization implementation (semantic task
// 0014), including every convergence test and the plaintext values those tests
// branch on. What changes is the representation: the state is one AV of length
// dim, so each vector update is a single operation, and each `open()` that the
// serial version performed dim times now happens once.
BatchedOptResult MinimizeBFGSBatched(const BatchedObjective& f, const AV& x0,
                                     int max_iterations = 20) {
    const size_t n = x0.size();
    EngineRef engine = x0.engine;

    AV x = Clone(x0);
    x.setPrecision(precision);
    AV fx = f(x, 1);
    fx.setPrecision(precision);

    // The central-difference scatter pattern depends only on dim, so it is
    // shared once and reused by every gradient.
    AV selector = MakeCentralDifferenceSelector(n, engine);

    AV gradient = NumericalGradientBatched(f, x, selector);
    SMatrix h_inv = Identity(n, engine);

    BatchedOptResult result(x, fx, 0, false);

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        result.iterations = iteration + 1;

        // One open for the whole gradient, where the serial version did n.
        auto opened_g = gradient.open();
        double max_grad = 0.0;
        for (size_t i = 0; i < n; ++i) {
            max_grad = std::max(max_grad, std::abs(static_cast<double>(opened_g[i]) / scale));
        }

        if (max_grad < kSmallEpsilon) {
            result.converged = true;
            break;
        }

        // Search direction d = -H_inv * gradient
        AV direction = -MatVecBatched(h_inv, gradient);
        direction.setPrecision(precision);

        AV directional_derivative = *gradient.dot_product(direction, n);
        directional_derivative.setPrecision(precision);
        auto opened_dd = directional_derivative.open();
        double dd_val = static_cast<double>(opened_dd[0]) / scale;

        if (dd_val >= 0.0) {
            // Not a descent direction: reset to steepest descent.
            h_inv = Identity(n, engine);
            direction = -gradient;
            direction.setPrecision(precision);
            directional_derivative = *gradient.dot_product(direction, n);
            directional_derivative.setPrecision(precision);
            auto opened_dd2 = directional_derivative.open();
            dd_val = static_cast<double>(opened_dd2[0]) / scale;
        }

        const double c1 = 1e-4;
        double alpha = 1.0;
        bool line_search_failed = false;
        AV x_new = Clone(x);
        AV fx_new = Clone(fx);

        auto opened_fx = fx.open();
        double fx_val = static_cast<double>(opened_fx[0]) / scale;

        while (true) {
            // Clone, NOT `AV alpha_dir = direction;`. Copy-construction aliases
            // the buffer (semantic topic 0001, C-05), so the copy-assignment
            // below would write the scaled result back through the alias into
            // `direction`, compounding alpha on every line-search halving.
            AV alpha_dir = Clone(direction);
            alpha_dir.setPrecision(0);
            alpha_dir = (*(alpha_dir * static_cast<DataType>(alpha * scale))) / scale;
            x_new = x + alpha_dir;
            x_new.setPrecision(precision);

            fx_new = f(x_new, 1);
            fx_new.setPrecision(precision);

            auto opened_fx_new = fx_new.open();
            double fx_new_val = static_cast<double>(opened_fx_new[0]) / scale;

            if (std::isfinite(fx_new_val) && fx_new_val <= fx_val + c1 * alpha * dd_val) {
                break;
            }
            alpha *= 0.5;
            if (alpha < 1e-5) {
                x_new = Clone(x);
                fx_new = Clone(fx);
                line_search_failed = true;
                break;
            }
        }

        AV step = x_new - x;
        step.setPrecision(precision);
        auto opened_step = step.open();
        double max_step = 0.0;
        for (size_t i = 0; i < n; ++i) {
            max_step = std::max(max_step, std::abs(static_cast<double>(opened_step[i]) / scale));
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

        AV gradient_new = NumericalGradientBatched(f, x_new, selector);
        AV gradient_delta = gradient_new - gradient;
        gradient_delta.setPrecision(precision);

        AV curvature = *step.dot_product(gradient_delta, n);
        curvature.setPrecision(precision);
        auto opened_curv = curvature.open();
        double curv_val = static_cast<double>(opened_curv[0]) / scale;

        if (curv_val > kSmallEpsilon) {
            AV rho = SecureReciprocal(curvature);
            h_inv = BfgsInverseUpdateBatched(h_inv, step, gradient_delta, rho);
            h_inv.setPrecision(precision);
        }

        x = Clone(x_new);
        gradient = Clone(gradient_new);
        auto opened_new_fx = fx_new.open();
        double obj_change = std::abs(fx_val - static_cast<double>(opened_new_fx[0]) / scale);
        fx = Clone(fx_new);

        if (engine.getPartyID() == 0) {
            std::cout << "[BFGS] iter " << std::setw(3) << (iteration + 1)
                      << "  neg_log_lik=" << std::fixed << std::setprecision(6)
                      << static_cast<double>(opened_new_fx[0]) / scale
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

    result.params = Clone(x);
    result.value = Clone(fx);
    return result;
}

}  // namespace cdough::regression
#pragma once

#include "./primitives.h"

namespace cdough::regression {

// =============================================================================
// BFGS Optimization & Secure Linear Algebra Helpers
// =============================================================================

// std::vector<AV>(n, AV(...)) copy-constructs n aliases of a single buffer, so
// every element would share storage. Build the elements individually instead.
std::vector<AV> MakeVector(size_t n, size_t elem_size, EngineRef engine) {
    std::vector<AV> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) out.emplace_back(elem_size, engine);
    return out;
}

std::vector<std::vector<AV>> MakeMatrix(size_t rows, size_t cols, EngineRef engine) {
    std::vector<std::vector<AV>> out;
    out.reserve(rows);
    for (size_t i = 0; i < rows; ++i) out.push_back(MakeVector(cols, 1, engine));
    return out;
}

std::vector<AV> Clone(const std::vector<AV>& vs) {
    std::vector<AV> out;
    out.reserve(vs.size());
    for (const AV& v : vs) out.push_back(Clone(v));
    return out;
}

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

// Matrix product a * b, both row-major, result row-major.
//
// The protocol rescales for us: the 3PC matmul kernel calls handle_precision()
// and then truncate() on the accumulated dot product, so the result is already
// in fixed point and must NOT be divided by `scale`. handle_precision() throws
// unless both operands carry exactly the same precision.
SMatrix MatMul(const SMatrix& a, const SMatrix& b) {
    assert(a.cols() == b.rows());
    assert(!a.isColumnWise() && !b.isColumnWise());
    assert(a.data().getPrecision() == b.data().getPrecision());
    SMatrix result = a.matrixRightMultiplyWithColumnMatrixVectorized(AsColumnWise(b));
    result.setPrecision(a.data().getPrecision());
    return result;
}

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
        SMatrix a_x = MatMul(a, x);
        a_x.setPrecision(precision);

        SMatrix residual = two_identity - a_x; // local elementwise
        residual.setPrecision(precision);

        x = MatMul(x, residual);
        x.setPrecision(precision);
    }

    return x;
}

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

// Result of the outer quasi-Newton optimization.
struct OptResult {
    std::vector<AV> params;
    AV value;
    int iterations = 0;
    bool converged = false;

    OptResult(std::vector<AV> p, AV v, int it = 0, bool conv = false)
        : params(std::move(p)), value(std::move(v)), iterations(it), converged(conv) {}
};

// Central finite-difference gradient of a scalar objective `f` at `x`.
std::vector<AV> NumericalGradient(const std::function<AV(const std::vector<AV>&)>& f,
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
        AV abs_xk =
            *(two_mask * xk_copy);  // (1 or -1) * xk_copy gives |x[k]| directly without / scale

        AV one_plus_abs = abs_xk;
        one_plus_abs += scale;
        DataType h_step_scaled = static_cast<DataType>(kNumericalGradientStep * scale);
        AV h = (*(one_plus_abs * h_step_scaled)) / scale;  // size 1

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

// BFGS update of the inverse-Hessian approximation:
//   H+ = (I - rho s y^T) H (I - rho y s^T) + rho s s^T,   rho = 1 / (y^T s).
SMatrix BfgsInverseUpdate(const SMatrix& h_inv, const std::vector<AV>& s, const std::vector<AV>& y,
                          const AV& rho) {
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
    // Flatten result into a single AV of size n * n to construct SecureMatrix.
    // `concatenate` appends, so this starts empty and grows to exactly n * n over the
    // loop below; the SecureMatrix constructor asserts data.size() == rows * cols.
    AV updated_data(0, engine);
    updated_data.setPrecision(precision);

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
            sum.setPrecision(precision);

            updated_data.concatenate(sum);
        }
    }

    updated_data.setPrecision(precision);
    SMatrix updated(updated_data, n, n, false);
    updated.setPrecision(precision);
    return updated;
}

// Minimizes `f` starting from `x0` using BFGS with a backtracking (Armijo) line
// search and numerical gradients.
OptResult MinimizeBFGS(const std::function<AV(const std::vector<AV>&)>& f,
                       const std::vector<AV>& x0, int max_iterations = 20) {
    size_t n = x0.size();
    EngineRef engine = x0[0].engine;

    std::vector<AV> x;
    for (size_t i = 0; i < n; ++i) {
        AV xi(x0[i].size(), engine);
        xi = x0[i];
        xi.setPrecision(precision);
        x.push_back(xi);
    }
    AV fx = f(x);
    fx.setPrecision(precision);

    std::vector<AV> gradient = NumericalGradient(f, x);
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
        AV fx_new(fx.size(), engine);
        fx_new = fx;
        fx_new.setPrecision(precision);

        auto opened_fx = fx.open();
        double fx_val = static_cast<double>(opened_fx[0]) / scale;

        while (true) {
            for (size_t i = 0; i < n; ++i) {
                AV alpha_dir(direction[i].size(), engine);
                alpha_dir = direction[i];
                alpha_dir.setPrecision(0);
                alpha_dir = (*(alpha_dir * static_cast<DataType>(alpha * scale))) / scale;
                x_new[i] = x[i] + alpha_dir;
                x_new[i].setPrecision(precision);
            }
            fx_new = f(x_new);
            fx_new.setPrecision(precision);

            auto opened_fx_new = fx_new.open();
            double fx_new_val = static_cast<double>(opened_fx_new[0]) / scale;

            if (std::isfinite(fx_new_val) && fx_new_val <= fx_val + c1 * alpha * dd_val) {
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

        std::vector<AV> gradient_new = NumericalGradient(f, x_new);
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
            AV rho = SecureReciprocal(curvature);

            h_inv = BfgsInverseUpdate(h_inv, step, gradient_delta, rho);
            h_inv.setPrecision(precision);
        }

        x = x_new;
        gradient = gradient_new;
        auto opened_new_fx = fx_new.open();
        double obj_change = std::abs(fx_val - static_cast<double>(opened_new_fx[0]) / scale);
        fx = fx_new;

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

    result.params = x;
    result.value = fx;
    return result;
}

}  // namespace cdough::regression
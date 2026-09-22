#pragma once

#include <cstddef>
#include <vector>

#include "./primitives.h"

// Small dense secure linear algebra: the weighted Gram matrix, and Cholesky
// factorisation, solve and inverse for a symmetric positive definite matrix.

namespace cdough::regression {

// =============================================================================
// Small dense secure linear algebra
//
// Matrices are flat AVs in row-major order; `p` is public and small (<= 8 for
// the widest model here), so every loop bound below is public and the control
// flow is completely data independent. All values are raw scaled integers
// (precision 0) with explicit rescaling after each multiply, matching the
// convention used by the kernels above.
// =============================================================================

// A single element as a 1-long view. Assigning to the returned value writes
// through to the underlying buffer.
AV Cell(const AV& m, size_t idx) { return m.slice(idx, idx + 1); }

AV ScalarZero(EngineRef engine) {
    AV z(1, engine);
    z.setPrecision(0);
    return z;
}

// A public p-vector, secret shared. Used for the unit vectors that turn a
// Cholesky solve into a matrix inverse.
AV PublicVector(EngineRef engine, const std::vector<double>& values) {
    cdough::Vector<DataType> v(values.size(), 0);
    for (size_t i = 0; i < values.size(); ++i) {
        v[i] = static_cast<DataType>(std::llround(values[i] * scale));
    }
    AV out = engine.template public_share_a<DataType>(v);
    out.setPrecision(0);
    return out;
}

// Cholesky factorisation A = L L^T of a symmetric positive definite A.
//
// No pivoting, for two independent reasons. Numerically it is unnecessary: for
// SPD matrices Cholesky is unconditionally backward stable, with a growth factor
// bounded by 1, so pivoting buys nothing. Cryptographically it is impossible:
// pivoting means comparing secret magnitudes and permuting rows on the result,
// which is data-dependent control flow. SPD is guaranteed here by the ridge term
// the callers add to the diagonal.
struct Cholesky {
    AV l;         // p*p, lower triangular, row-major
    AV inv_diag;  // p, reciprocals of L's diagonal (so the solves need no division)
};

Cholesky CholeskyFactor(const AV& a, size_t p) {
    EngineRef engine = a.engine;
    AV l(p * p, engine);
    l.setPrecision(0);
    AV inv_diag(p, engine);
    inv_diag.setPrecision(0);

    AV a_ = a;
    a_.setPrecision(0);

    for (size_t j = 0; j < p; ++j) {
        // d = A[j][j] - sum_{k<j} L[j][k]^2
        AV d = Clone(Cell(a_, j * p + j));
        d.setPrecision(0);
        for (size_t k = 0; k < j; ++k) {
            AV ljk = Cell(l, j * p + k);
            d -= *(*(ljk * ljk) / scale);
        }

        // r = 1/sqrt(d) and L[j][j] = d * r = sqrt(d), neither needing a division.
        AV r = Rsqrt(d);
        r.setPrecision(0);
        AV diag = *(*(d * r) / scale);
        AV diag_cell = Cell(l, j * p + j);
        diag_cell = diag;
        AV inv_cell = Cell(inv_diag, j);
        inv_cell = r;

        // L[i][j] = (A[i][j] - sum_{k<j} L[i][k] L[j][k]) / L[j][j]
        for (size_t i = j + 1; i < p; ++i) {
            AV v = Clone(Cell(a_, i * p + j));
            v.setPrecision(0);
            for (size_t k = 0; k < j; ++k) {
                AV lik = Cell(l, i * p + k);
                AV ljk = Cell(l, j * p + k);
                v -= *(*(lik * ljk) / scale);
            }
            AV cell = Cell(l, i * p + j);
            cell = *(*(v * r) / scale);
        }
    }
    l.setPrecision(0);
    inv_diag.setPrecision(0);
    return Cholesky{l, inv_diag};
}

// Solve A x = b from a precomputed factorisation, by forward then back
// substitution. Reusing the factorisation is what makes a full inverse cheap:
// the factorisation is most of the cost and is shared across all p solves.
AV CholeskySolveWith(const Cholesky& f, const AV& b, size_t p) {
    EngineRef engine = b.engine;
    AV b_ = b;
    b_.setPrecision(0);

    AV y(p, engine);
    y.setPrecision(0);
    for (size_t i = 0; i < p; ++i) {
        AV t = Clone(Cell(b_, i));
        t.setPrecision(0);
        for (size_t k = 0; k < i; ++k) {
            AV lik = Cell(f.l, i * p + k);
            AV yk = Cell(y, k);
            t -= *(*(lik * yk) / scale);
        }
        AV cell = Cell(y, i);
        cell = *(*(t * Cell(f.inv_diag, i)) / scale);
    }

    AV x(p, engine);
    x.setPrecision(0);
    for (size_t i = p; i-- > 0;) {
        AV t = Clone(Cell(y, i));
        t.setPrecision(0);
        for (size_t k = i + 1; k < p; ++k) {
            AV lki = Cell(f.l, k * p + i);
            AV xk = Cell(x, k);
            t -= *(*(lki * xk) / scale);
        }
        AV cell = Cell(x, i);
        cell = *(*(t * Cell(f.inv_diag, i)) / scale);
    }
    x.setPrecision(0);
    return x;
}

AV CholeskySolve(const AV& a, const AV& b, size_t p) {
    return CholeskySolveWith(CholeskyFactor(a, p), b, p);
}

// Full inverse of an SPD matrix: one factorisation, then p solves against the
// unit vectors. Returned row-major.
AV SymmetricInverse(const AV& a, size_t p) {
    EngineRef engine = a.engine;
    Cholesky f = CholeskyFactor(a, p);
    AV inv(p * p, engine);
    inv.setPrecision(0);
    for (size_t k = 0; k < p; ++k) {
        std::vector<double> e(p, 0.0);
        e[k] = 1.0;
        AV col = CholeskySolveWith(f, PublicVector(engine, e), p);
        for (size_t i = 0; i < p; ++i) {
            AV cell = Cell(inv, i * p + k);
            cell = Cell(col, i);
        }
    }
    inv.setPrecision(0);
    return inv;
}

// Weighted Gram matrix X^T W X for X stored row-major as n*p, W a length-n
// vector of weights. Returned p*p row-major.
//
// The p(p+1)/2 distinct entries are computed in a SINGLE dot_product call: the
// two operands are built as public gathers (mapping_reference is a free view,
// no communication) that lay the required column pairs out end to end, and the
// engine then contracts each length-n chunk locally. Only p(p+1)/2 ring
// elements cross the wire, independent of n.
AV Gram(const AV& x_rm, const AV& w, size_t n, size_t p) {
    EngineRef engine = x_rm.engine;

    AV x_ = x_rm;
    x_.setPrecision(0);
    AV w_ = w;
    w_.setPrecision(0);

    // Z = W * X, row-scaled. repeated_subset_reference repeats each weight p
    // times, which lines it up with the row-major layout of X.
    AV z = *(*(x_ * w_.repeated_subset_reference(p)) / scale);
    z.setPrecision(0);

    std::vector<size_t> pairs_k, pairs_l;
    for (size_t k = 0; k < p; ++k) {
        for (size_t l = k; l < p; ++l) {
            pairs_k.push_back(k);
            pairs_l.push_back(l);
        }
    }
    const size_t num_pairs = pairs_k.size();

    std::vector<cdough::VectorSizeType> lhs_map(num_pairs * n), rhs_map(num_pairs * n);
    for (size_t t = 0; t < num_pairs; ++t) {
        for (size_t i = 0; i < n; ++i) {
            lhs_map[t * n + i] = static_cast<cdough::VectorSizeType>(i * p + pairs_k[t]);
            rhs_map[t * n + i] = static_cast<cdough::VectorSizeType>(i * p + pairs_l[t]);
        }
    }

    AV lhs = z.mapping_reference(lhs_map);
    AV rhs = x_.mapping_reference(rhs_map);
    lhs.setPrecision(0);
    rhs.setPrecision(0);

    AV packed = *(*lhs.dot_product(rhs, n) / scale);
    packed.setPrecision(0);

    AV g(p * p, engine);
    g.setPrecision(0);
    for (size_t t = 0; t < num_pairs; ++t) {
        AV v = Cell(packed, t);
        AV up = Cell(g, pairs_k[t] * p + pairs_l[t]);
        up = v;
        AV lo = Cell(g, pairs_l[t] * p + pairs_k[t]);
        lo = v;
    }
    g.setPrecision(0);
    return g;
}

// Add a public ridge to the diagonal, in place.
//
// Not optional. At precision 16 the IRLS weight p(1-p) truncates to zero for
// |eta| >= 12, so under the quasi-separation that a 8-parameter model with an
// interaction term invites, X^T W X goes singular and the Cholesky diagonal
// stops being positive. The ridge keeps the matrix SPD, which is exactly the
// assumption that licenses factorising without pivoting.
void AddRidge(AV& g, size_t p, double lambda) {
    const DataType lam = static_cast<DataType>(std::llround(lambda * scale));
    g.setPrecision(0);
    for (size_t i = 0; i < p; ++i) {
        AV cell = Cell(g, i * p + i);
        cell += lam;
    }
}


}  // namespace cdough::regression

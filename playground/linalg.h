#pragma once

#include <cstddef>
#include <vector>

#include "./library/primitives.h"

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

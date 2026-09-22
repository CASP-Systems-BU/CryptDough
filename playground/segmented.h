#pragma once

#include <cassert>
#include <cstddef>
#include <tuple>
#include <vector>

#include "./primitives.h"

// Per-group (segmented) scans over a table sorted by its grouping key, and the
// group-boundary indicators built on them.

namespace cdough::regression {

// =============================================================================
// Segmented (per-group) helpers
//
// Everything in this section assumes the rows are SORTED by `keys`, padded to a
// power of two (aggregate() asserts it, aggregation.h:198), that the pad rows
// carry a key sentinel which cannot collide with a real group, and that the pad
// rows of every value column are zero. aggregators::aggregate has no valid-bit
// handling of its own -- that all lives in EncodedTable -- so the caller owns
// those invariants.
// =============================================================================

// common.h:14 declares `enum class Direction { ... } Direction;` -- that trailing
// name is a VARIABLE which shadows the type, so the type can only be named with
// an elaborated specifier. aggregate() itself works around this the same way
// (`const enum Direction dir`).
using SegDirection = enum cdough::aggregators::Direction;

using AggSpecA = std::vector<std::tuple<AV, AV, void (*)(const AV&, AV&, const AV&)>>;
using AggSpecB = std::vector<std::tuple<BV, BV, void (*)(const BV&, BV&, const BV&)>>;

size_t NextPowerOfTwo(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Per-group inclusive scan. Forward leaves each row holding the sum of its
// group up to and including itself; Reverse does the same from the far end.
void SegScan(std::vector<BV>& keys, const std::vector<AV>& in, std::vector<AV>& out,
             SegDirection dir) {
    assert(in.size() == out.size());
    AggSpecB b_spec;
    AggSpecA a_spec;
    a_spec.reserve(in.size());
    // The aggregator computes `a + group * b`, where the group bit is an
    // arithmetic 0/1 at precision 0. handle_precision (protocol.h:115) throws on
    // a precision mismatch, so every column here is handled as a raw scaled
    // integer; inputs and outputs are precision 0 in and precision 0 out.
    std::vector<AV> in_p0, out_p0;
    in_p0.reserve(in.size());
    out_p0.reserve(in.size());
    for (size_t k = 0; k < in.size(); ++k) {
        in_p0.push_back(in[k]);
        in_p0.back().setPrecision(0);
        out_p0.push_back(out[k]);
        out_p0.back().setPrecision(0);
    }
    for (size_t k = 0; k < in.size(); ++k) {
        a_spec.push_back({in_p0[k], out_p0[k], &cdough::aggregators::sum<AV>});
    }
    cdough::aggregators::aggregate(keys, b_spec, a_spec, dir);
    for (size_t k = 0; k < out.size(); ++k) out[k].setPrecision(0);
}

// Broadcast each group's TOTAL to every row of that group.
//
// The obvious formulation -- scan forward, then scan the result backwards -- is
// wrong: a Reverse pass over an already-scanned column produces a
// suffix-of-prefixes, which is silently garbage rather than an error. Both
// passes must read the ORIGINAL column, and the row's own value is then double
// counted exactly once:
//
//     total = prefix + suffix - self
//
// The subtraction is local, so this costs two aggregate invocations and no
// extra communication.
void SegTotal(std::vector<BV>& keys, const std::vector<AV>& in, std::vector<AV>& out) {
    assert(in.size() == out.size());
    const size_t n = in[0].size();

    std::vector<AV> suffix;
    suffix.reserve(in.size());
    for (size_t k = 0; k < in.size(); ++k) suffix.emplace_back(n, in[k].engine);

    SegScan(keys, in, out, SegDirection::Forward);
    SegScan(keys, in, suffix, SegDirection::Reverse);

    for (size_t k = 0; k < in.size(); ++k) {
        AV self = Clone(in[k]);
        self.setPrecision(0);
        out[k].setPrecision(0);
        suffix[k].setPrecision(0);
        out[k] += suffix[k];
        out[k] -= self;
    }
}

// Single-column convenience wrappers.
AV SegTotal(std::vector<BV>& keys, const AV& in) {
    std::vector<AV> ins{in};
    std::vector<AV> outs;
    outs.emplace_back(in.size(), in.engine);
    SegTotal(keys, ins, outs);
    return outs[0];
}

AV SegScan(std::vector<BV>& keys, const AV& in, SegDirection dir) {
    std::vector<AV> ins{in};
    std::vector<AV> outs;
    outs.emplace_back(in.size(), in.engine);
    SegScan(keys, ins, outs, dir);
    return outs[0];
}

bool g_use_cached_scans = true;

struct ScanPlan {
    size_t n = 0;
    int depth = 0;
    // Arithmetic 0/1 "same group" bits, one vector per network level, for each
    // scan direction.
    std::vector<AV> forward;
    std::vector<AV> reverse;
};

// The (earlier, later) index pair the Brent-Kung network compares at one level.
struct ScanLevel {
    size_t a_start, b_start, step, count;
};

ScanLevel PlanLevel(size_t n, int level) {
    const int half = static_cast<int>(std::bit_width(n - 1));
    size_t gap, a_start, b_start;
    if (level < half) {
        gap = size_t{1} << level;
        a_start = gap - 1;
        b_start = 2 * gap - 1;
    } else {
        const int mirrored = 2 * half - 2 - level;
        gap = size_t{1} << mirrored;
        a_start = 2 * gap - 1;
        b_start = 3 * gap - 1;
    }
    const size_t step = 2 * gap;
    // `a` may run further than `b`; the network truncates it to b's length.
    const size_t count = b_start <= n - 1 ? (n - 1 - b_start) / step + 1 : 0;
    return ScanLevel{a_start, b_start, step, count};
}

int ScanDepth(size_t n) { return 2 * static_cast<int>(std::bit_width(n - 1)) - 1; }

// A write-through reversed view, used to run the forward geometry backwards.
std::vector<cdough::VectorSizeType> ReverseMap(size_t n) {
    std::vector<cdough::VectorSizeType> map(n);
    for (size_t i = 0; i < n; ++i) map[i] = static_cast<cdough::VectorSizeType>(n - 1 - i);
    return map;
}

ScanPlan BuildScanPlan(const BV& key) {
    EngineRef engine = key.engine;
    ScanPlan plan;
    plan.n = key.size();
    plan.depth = ScanDepth(plan.n);

    // A materialised reversed copy of the keys, so the reverse direction can use
    // the same level arithmetic.
    BV key_rev(plan.n, engine);
    key_rev = key.mapping_reference(ReverseMap(plan.n));

    for (int pass = 0; pass < 2; ++pass) {
        const BV& k = (pass == 0) ? key : key_rev;
        std::vector<AV>& into = (pass == 0) ? plan.forward : plan.reverse;
        for (int level = 0; level < plan.depth; ++level) {
            const ScanLevel geom = PlanLevel(plan.n, level);
            if (geom.count == 0) {
                into.emplace_back(1, engine);
                continue;
            }
            BV a = k.simple_subset_reference(geom.a_start, geom.step,
                                             geom.a_start + (geom.count - 1) * geom.step);
            BV b = k.simple_subset_reference(geom.b_start, geom.step,
                                             geom.b_start + (geom.count - 1) * geom.step);
            AV same = *((a == b)->b2a_bit());
            same.setPrecision(0);
            into.push_back(same);
        }
    }
    return plan;
}

// Multi-key plan: the per-level "same group" bit is the AND over every key
// column of that column's equality. The bits are arithmetic 0/1, so the AND is
// a multiply -- one extra multiply per level per extra key, paid once when the
// plan is built and then reused by every scan over that key.
//
// This is what lets the per-system re-sequencing keep the cached fast path: its
// group key is the compound {subject_id, data_source}, not a single column.
ScanPlan BuildScanPlan(const std::vector<BV>& keys) {
    assert(!keys.empty());
    if (keys.size() == 1) return BuildScanPlan(keys[0]);

    EngineRef engine = keys[0].engine;
    ScanPlan plan;
    plan.n = keys[0].size();
    plan.depth = ScanDepth(plan.n);

    const std::vector<cdough::VectorSizeType> rev = ReverseMap(plan.n);
    std::vector<BV> key_rev;
    key_rev.reserve(keys.size());
    for (const BV& k : keys) {
        BV r(plan.n, engine);
        r = k.mapping_reference(rev);
        key_rev.push_back(r);
    }

    for (int pass = 0; pass < 2; ++pass) {
        const std::vector<BV>& ks = (pass == 0) ? keys : key_rev;
        std::vector<AV>& into = (pass == 0) ? plan.forward : plan.reverse;
        for (int level = 0; level < plan.depth; ++level) {
            const ScanLevel geom = PlanLevel(plan.n, level);
            if (geom.count == 0) {
                into.emplace_back(1, engine);
                continue;
            }
            AV same(geom.count, engine);
            for (size_t c = 0; c < ks.size(); ++c) {
                BV a = ks[c].simple_subset_reference(
                    geom.a_start, geom.step, geom.a_start + (geom.count - 1) * geom.step);
                BV b = ks[c].simple_subset_reference(
                    geom.b_start, geom.step, geom.b_start + (geom.count - 1) * geom.step);
                AV eq = *((a == b)->b2a_bit());
                eq.setPrecision(0);
                if (c == 0) {
                    same = eq;
                } else {
                    AV both = *(same * eq);
                    both.setPrecision(0);
                    same = both;
                }
                same.setPrecision(0);
            }
            into.push_back(same);
        }
    }
    return plan;
}

// Per-group inclusive scan using a precomputed plan. Same contract as SegScan.
void SegScanPlanned(const ScanPlan& plan, const std::vector<AV>& in, std::vector<AV>& out,
                    SegDirection dir) {
    assert(in.size() == out.size());
    const size_t n = plan.n;
    const std::vector<AV>& bits =
        (dir == SegDirection::Forward) ? plan.forward : plan.reverse;
    const bool reversed = (dir != SegDirection::Forward);
    const std::vector<cdough::VectorSizeType> rev = reversed ? ReverseMap(n)
                                                             : std::vector<cdough::VectorSizeType>{};

    for (size_t k = 0; k < in.size(); ++k) {
        out[k].setPrecision(0);
        AV src = in[k];
        src.setPrecision(0);
        out[k] = src;  // deep element-wise copy, as aggregate() also does

        // Reversing the accumulator lets one set of index arithmetic serve both
        // directions; the mapping is a public view, so it writes through.
        AV acc = reversed ? out[k].mapping_reference(rev) : out[k];
        acc.setPrecision(0);

        for (int level = 0; level < plan.depth; ++level) {
            const ScanLevel geom = PlanLevel(n, level);
            if (geom.count == 0) continue;
            AV a = acc.simple_subset_reference(geom.a_start, geom.step,
                                               geom.a_start + (geom.count - 1) * geom.step);
            AV b = acc.simple_subset_reference(geom.b_start, geom.step,
                                               geom.b_start + (geom.count - 1) * geom.step);
            a.setPrecision(0);
            b.setPrecision(0);
            b += *(bits[level] * a);  // later += same_group * earlier
        }
        out[k].setPrecision(0);
    }
}

// Broadcast each group's total to every row, using a precomputed plan.
void SegTotalPlanned(const ScanPlan& plan, const std::vector<AV>& in, std::vector<AV>& out) {
    const size_t n = plan.n;
    std::vector<AV> suffix;
    suffix.reserve(in.size());
    for (size_t k = 0; k < in.size(); ++k) suffix.emplace_back(n, in[k].engine);

    SegScanPlanned(plan, in, out, SegDirection::Forward);
    SegScanPlanned(plan, in, suffix, SegDirection::Reverse);

    for (size_t k = 0; k < in.size(); ++k) {
        AV self = Clone(in[k]);
        self.setPrecision(0);
        out[k].setPrecision(0);
        suffix[k].setPrecision(0);
        out[k] += suffix[k];
        out[k] -= self;
    }
}

// Single-column convenience wrapper for the planned scan, mirroring the
// SegScan/SegTotal pair above.
AV SegScanPlanned(const ScanPlan& plan, const AV& in, SegDirection dir) {
    std::vector<AV> ins{in};
    std::vector<AV> outs;
    outs.emplace_back(in.size(), in.engine);
    SegScanPlanned(plan, ins, outs, dir);
    return outs[0];
}

AV SegTotalPlanned(const ScanPlan& plan, const AV& in) {
    std::vector<AV> ins{in};
    std::vector<AV> outs;
    outs.emplace_back(in.size(), in.engine);
    SegTotalPlanned(plan, ins, outs);
    return outs[0];
}

// 1 on the first row of each run of equal keys (operators::distinct marks
// exactly that, and always marks row 0).
BV FirstOfGroup(std::vector<BV>& keys) {
    BV uniq(keys[0].size(), keys[0].engine);
    cdough::operators::distinct(keys, uniq);
    return uniq;
}

// 1 on the last row of each run of equal keys: last[i] = first[i+1], with the
// final row always last. This is the same shift EncodedTable::aggregate uses to
// pick the surviving row of a group (encoded_table.h:1311).
BV LastOfGroup(std::vector<BV>& keys) {
    const size_t n = keys[0].size();
    BV uniq = FirstOfGroup(keys);

    BV last(n, keys[0].engine);
    last.zero();
    BV head = last.slice(0, n - 1);
    head = uniq.slice(1);

    cdough::Vector<DataType> one(1, 1);
    BV tail = last.slice(n - 1);
    tail = keys[0].engine.template public_share_b<DataType>(one);
    return last;
}

// Arithmetic 0/1 indicator of the last row of each group, which is the mask that
// makes a per-cluster quantity count exactly once.
AV LastOfGroupArith(std::vector<BV>& keys) {
    BV last = LastOfGroup(keys);
    AV out = *(last.b2a_bit());
    out.setPrecision(0);
    return out;
}

// =============================================================================
// Oblivious rank
//
// ROW_NUMBER() OVER (PARTITION BY keys ORDER BY <the order the rows already
// sit in>). The table must ALREADY be sorted by (key, time) -- ranking does not
// order anything, it only counts -- and the rank is then a segmented prefix sum
// over a column of ones.
// =============================================================================

AV SegRank(std::vector<BV>& keys, const AV& ones, SegDirection dir = SegDirection::Forward) {
    AV out = SegScan(keys, ones, dir);
    out.setPrecision(0);
    return out;
}

AV SegRankPlanned(const ScanPlan& plan, const AV& ones,
                  SegDirection dir = SegDirection::Forward) {
    AV out = SegScanPlanned(plan, ones, dir);
    out.setPrecision(0);
    return out;
}

// FIRST_VALUE(col) OVER (PARTITION BY keys ORDER BY ...) -- the value on each
// group's first row, broadcast to every row of that group. `first` is the
// arithmetic first-of-group indicator; zeroing every other row and taking the
// group total leaves that one value everywhere. Used for index_dt, from which
// the follow-up bands are measured.
AV FirstValueSeed(const AV& col, const AV& first) {
    AV c = Clone(col);
    c.setPrecision(0);
    AV f = Clone(first);
    f.setPrecision(0);
    AV seed = *(c * f);
    seed.setPrecision(0);
    return seed;
}

AV SegFirstValue(std::vector<BV>& keys, const AV& col, const AV& first) {
    AV seed = FirstValueSeed(col, first);
    return SegTotal(keys, seed);
}

AV SegFirstValuePlanned(const ScanPlan& plan, const AV& col, const AV& first) {
    AV seed = FirstValueSeed(col, first);
    return SegTotalPlanned(plan, seed);
}

// COUNT(DISTINCT key) over the rows where `mask` is 1.
//
// The table is already sorted by key, so a group contributes iff the mask is set
// anywhere inside it. SegTotal gives the per-group mask count on every row;
// gtez on (count - 1) turns that into "seen at least once"; and multiplying by
// the first-of-group indicator counts each group exactly once.
AV CountDistinct(std::vector<BV>& keys, const AV& mask) {
    AV seen = SegTotal(keys, mask);
    seen.setPrecision(0);
    seen -= DataType(1);
    AV any = *(seen.gtez());  // 1 if the group has at least one masked row

    BV first_b = FirstOfGroup(keys);
    AV first = *(first_b.b2a_bit());
    first.setPrecision(0);

    AV contrib = *(any * first);
    AV total = contrib.chunkedSum(contrib.size());
    total.setPrecision(0);
    return total;
}

}  // namespace cdough::regression

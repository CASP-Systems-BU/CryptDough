#include "cdough.h"

using namespace cdough::debug;
using namespace cdough::service;
using namespace cdough::aggregators;
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

template <typename T>
auto construct_like(const T& x, cdough::VectorSizeType size) {
    if constexpr (std::is_base_of<cdough::EncodedVector, T>::value) {
        return T(size, x.engine);
    } else {
        return T(size);
    }
}

template <typename T, typename Engine>
auto bench_prefix_sum_internal(T x, Engine& engine) {
    auto n = x.size();

    // TODO: deprecated constructor usage in public API
    auto w = construct_like(x, n);
    auto y = construct_like(x, n);
    auto z = construct_like(x, n);

    w = x;
    y = x;
    z = x;

    // T yi(1), yp(1);

    stopwatch::timepoint("--");
    w.prefix_sum();
    stopwatch::timepoint("Direct EVector");

    engine.worker0->proto_32->mark_statistics();
    for (int i = 1; i < y.size(); i++) {
        // currently xi
        auto yi = y.slice(i, i + 1);
        auto yp = y.slice(i - 1, i);
        yi += yp;
    }
    stopwatch::timepoint("AP Linear");
    engine.print_statistics();

    engine.worker0->proto_32->mark_statistics();
    tree_prefix_sum(z);
    stopwatch::timepoint("Tree");
    engine.print_statistics();

    Vector<int> z_(x.size());
    if constexpr (std::is_base_of<cdough::EncodedVector, T>::value) {
        z_ = z.open();
        assert(z_.same_as(y.open()));
        assert(z_.same_as(w.open()));
    } else {
        z_ = z;
        assert(z.same_as(y));
        assert(z.same_as(w));
    }

    return z_;
}

template <typename Engine>
void bench_prefix_sum(int i, Engine& engine) {
    auto N = 1 << i;
    single_cout("Benchmark size 2^" << i << " (" << N << ")");
    auto pid = engine.getPartyID();

    Vector<int> x(N);
    engine.populateLocalRandom(x);
    Vector<int> y(N);
    y = x;

    single_cout("== Plaintext ==");

    // Plaintext
    if (pid == 0) {
        bench_prefix_sum_internal(x, engine);
    }

    single_cout("== AShared ==");

    ASharedVector<int> ash_x = engine.secret_share_a(x, 0);
    // TODO: deprecated constructor usage in public API
    ASharedVector<int> pf(ash_x.size(), engine);
    ASharedVector<int> sum(1, engine);

    stopwatch::timepoint("Begin");

    for (int j = 0; j < ash_x.vector.replicationNumber; j++) {
        for (int i = 0; i < ash_x.size(); i++) {
            sum.vector(j)[0] += ash_x.vector(j)[i];
            pf.vector(j)[i] = sum.vector(j)[0];
        }
    }

    stopwatch::timepoint("Manual");

    bench_prefix_sum_internal(ash_x, engine);

    // single_cout("== BShared ==");

    // BSharedVector<int> bsh_x = engine.secret_share_b(x, 0);
    // bench_prefix_sum_internal(bsh_x);
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);

    bench_prefix_sum(20, engine);
}
#include "cdough.h"
#include "profiling/stopwatch.h"

using namespace cdough::debug;
using namespace cdough::service;
using namespace cdough::random;
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
#ifndef MPC_PROTOCOL_BEAVER_TWO
    single_cout("Skipping test_correlated for non-2PC");
#else

    auto pID = engine.getPartyID();
    auto test_size = engine.getArg<size_t>("test-size", "r", 1 << 20);

    // setup generators
    using OLEBase = cdough::random::OLEGenerator<int32_t>;
    using ole_t = OLEBase::ole_t;

    auto comm = engine.comm0();

    auto generator = make_pooled<GilboaOLE<int32_t> >(pID, comm, 0);
    auto btgen =
        std::make_shared<BeaverTripleGenerator<int32_t, cdough::Encoding::AShared> >(generator, comm);

    stopwatch::timepoint("Start");

    /*
        generate without pooling
    */
    btgen->getNext(test_size);
    stopwatch::timepoint("Without Pooling");

    /*
        generate with pooling
    */

    // generation phase
    engine.reserve_mul_triples<int32_t>(test_size);
    stopwatch::timepoint("Pooling Generation Phase");

    auto batch = btgen->getNext(test_size);
    stopwatch::timepoint("Pooling Retrieval Phase");

#endif

    return 0;
}
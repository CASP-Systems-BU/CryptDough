#include <numeric>

#include "cdough.h"

// ../scripts/run_experiment.py -p 3 -s same -c mpi -r 1 -T 1 micro_ml_secure

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;

using DataType = int64_t;
using HeightWidth = cdough::matrix::HeightWidth;

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();
    auto test_size = engine.getArg<size_t>("test-size", "r", 192);

    stopwatch::timepoint("Start");
    const size_t rows = 224;
    const size_t cols = 224;
    HeightWidth inputHW{rows, cols};
    const size_t inChannels = 64;
    const size_t outChannels = 64;
    const size_t filterSize = 3;
    HeightWidth filterHW{filterSize, filterSize * inChannels};
    HeightWidth strideHW{1, 1 * inChannels};
    HeightWidth paddingHW{1, 1 * inChannels};
    size_t weightsSize = 512;
    int iterCount = 4;

    auto inputPlain =
        PlainMatrix<DataType>::RandomMatrix(engine, test_size * rows, cols * inChannels);
    auto filterPlain = PlainMatrix<DataType>::RandomMatrix(engine, outChannels * filterSize,
                                                           filterSize * inChannels);
    auto weightsPlain =
        PlainMatrix<DataType>::RandomColumnMatrix(engine, cols * inChannels, weightsSize);
    auto biasPlain = PlainMatrix<DataType>::RandomMatrix(engine, 1, weightsSize);
    stopwatch::timepoint("Input Gen");

    auto input = engine.secret_share_matrix(inputPlain, 0);
    auto filter = engine.secret_share_matrix(filterPlain, 0);
    auto weights = engine.secret_share_matrix(weightsPlain, 0);
    auto bias = engine.secret_share_matrix(biasPlain, 0);
    stopwatch::timepoint("Input Secret Share");

    {
        for (int i = 0; i < iterCount; i++) {
            auto c2 = input.matrixRightMultiplyVectorized(weights);
        }
        stopwatch::timepoint("Secure Matrix Mult_V");
    }

    {
        for (int i = 0; i < iterCount; i++) {
            auto c_2 = input.conv2DVectorized(filter, test_size, filterHW, strideHW, paddingHW);
        }
        stopwatch::timepoint("Secure Conv2D_V");
    }

    {
        for (int i = 0; i < iterCount; i++) {
            auto c_2 = input.fullyConnectedVectorized(weights, bias);
        }
        stopwatch::timepoint("Secure FullyConnected_V");
    }

    {
        for (int i = 0; i < iterCount; i++) {
            auto c_4 = input.avgPoolingVectorized(test_size, inChannels, inputHW, filterHW,
                                                  strideHW, {0, 0});
        }
        stopwatch::timepoint("Secure AvgPooling_V");
    }

    {
        for (int i = 0; i < iterCount; i++) {
            auto c_4 = input.reLUVectorized();
        }
        stopwatch::timepoint("Secure ReLU_V");
    }

    return 0;
}
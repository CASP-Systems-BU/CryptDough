#include <numeric>

#include "cdough.h"

// ../scripts/run_experiment.py -p 1 -s same -c nocopy -r 1 -T 1 micro_ml_local

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

    auto input = PlainMatrix<DataType>::RandomMatrix(engine, test_size * rows, cols * inChannels);
    auto filter = PlainMatrix<DataType>::RandomMatrix(engine, outChannels * filterSize,
                                                      filterSize * inChannels);
    auto weights =
        PlainMatrix<DataType>::RandomColumnMatrix(engine, cols * inChannels, weightsSize);
    auto bias = PlainMatrix<DataType>::RandomMatrix(engine, 1, weightsSize);
    stopwatch::timepoint("Input Gen");

    {
        for (int i = 0; i < iterCount; i++) {
            auto c2 = input.matrixRightMultiplyVectorized(weights);
        }
        stopwatch::timepoint("Matrix Mult_V");
    }

    {
        for (int i = 0; i < iterCount; i++) {
            auto c_2 = input.conv2DVectorized(filter, test_size, filterHW, strideHW, paddingHW);
        }
        stopwatch::timepoint("Conv2D_V");
    }

    {
        for (int i = 0; i < iterCount; i++) {
            auto c_2 = input.fullyConnectedVectorized(weights, bias);
        }
        stopwatch::timepoint("FullyConnected_V");
    }

    {
        for (int i = 0; i < iterCount; i++) {
            auto c_4 = input.avgPoolingVectorized(test_size, inChannels, inputHW, filterHW,
                                                  strideHW, {0, 0});
        }
        stopwatch::timepoint("AvgPooling_V");
    }

    {
        for (int i = 0; i < iterCount; i++) {
            auto c_4 = input.reLUVectorized();
        }
        stopwatch::timepoint("ReLU_V");
    }

    return 0;
}
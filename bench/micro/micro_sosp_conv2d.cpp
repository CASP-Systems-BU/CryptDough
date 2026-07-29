#include <numeric>
#include "cdough.h"

using namespace cdough::debug;
using namespace cdough::service;

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

using DataType = int64_t;
using HW = cdough::matrix::HeightWidth;

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();
    auto test_size = engine.getArg<size_t>("test-size", "r", 512);
    if (pID == 0)
        std::cout << "Using test size: " << test_size << " and batch size: " << engine.getBatchSize() << std::endl;

    // input = test_size * 224 rows, 64*22 columns
    const auto inputSize = HW(224, 224);
    const size_t channels_num = 64;
    const size_t precision = 0;

    auto inputPlain = PlainMatrix<DataType>::RandomMatrix(engine, test_size * inputSize.first, channels_num * inputSize.second);
    auto input = engine.secret_share_matrix(inputPlain, 0);
    input.setPrecision(precision);

    // Single conv2D layer model
    cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine> model(engine, precision);

    // "layer": "cnn",
    // "input_hw": [224, 224],
    // "in_channels": 64,
    // "out_channels": 64,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(224, 224), 64, 64, HW(3, 3), HW(1, 1), HW(1, 1));

    // Forward Pass
    stopwatch::timepoint("Start");
    auto output = model.forward(input);
    stopwatch::timepoint("Forward");

    return 0;
}

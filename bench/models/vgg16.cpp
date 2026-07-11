#include <numeric>

#include "cdough.h"

// python3 ../scripts/run_experiment.py -p 3 vgg16

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;

using DataType = int64_t;
using HW = cdough::matrix::HeightWidth;

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();
    auto test_size = engine.getArg<size_t>("test-size", "r", 192);
    if (pID == 0)
        std::cout << "Using test size: " << test_size
                  << " and batch size: " << engine.getBatchSize() << std::endl;

    // Test Data
    const auto inputSize = HW(32, 32);
    const size_t channels_num = 3;
    const size_t precision = 8;

    auto inputPlain = PlainMatrix<DataType>::RandomMatrix(engine, test_size * inputSize.first,
                                                          channels_num * inputSize.second);
    auto input = engine.secret_share_matrix(inputPlain, 0);
    input.setPrecision(precision);

    // VGG16
    cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine> model(engine, precision);

    // "layer": "cnn",
    // "input_hw": [32, 32],
    // "in_channels": 3,
    // "out_channels": 64,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(32, 32), 3, 64, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "relu",
    // "input_dim": 65536
    model.reLULayer(65536);

    // "layer": "cnn",
    // "input_hw": [32, 32],
    // "in_channels": 64,
    // "out_channels": 64,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(32, 32), 64, 64, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "averagepool",
    // "input_hw": [32, 32],
    // "in_channels": 64,
    // "pool_hw": [2, 2],
    // "stride": 2
    model.avgPoolingLayer(HW(32, 32), 64, HW(2, 2), HW(2, 2), HW(0, 0));
    // "layer": "relu",
    // "input_dim": 16384
    model.reLULayer(16384);

    // "layer": "cnn",
    // "input_hw": [16, 16],
    // "in_channels": 64,
    // "out_channels": 128,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(16, 16), 64, 128, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "relu",
    // "input_dim": 32768
    model.reLULayer(32768);

    // "layer": "cnn",
    // "input_hw": [16, 16],
    // "in_channels": 128,
    // "out_channels": 128,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(16, 16), 128, 128, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "averagepool",
    // "input_hw": [16, 16],
    // "in_channels": 128,
    // "pool_hw": [2, 2],
    // "stride": 2
    model.avgPoolingLayer(HW(16, 16), 128, HW(2, 2), HW(2, 2), HW(0, 0));
    // "layer": "relu",
    // "input_dim": 8192
    model.reLULayer(8192);

    // "layer": "cnn",
    // "input_hw": [8, 8],
    // "in_channels": 128,
    // "out_channels": 256,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(8, 8), 128, 256, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "relu",
    // "input_dim": 16384
    model.reLULayer(16384);

    // "layer": "cnn",
    // "input_hw": [8, 8],
    // "in_channels": 256,
    // "out_channels": 256,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(8, 8), 256, 256, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "relu",
    // "input_dim": 16384
    model.reLULayer(16384);

    // "layer": "cnn",
    // "input_hw": [8, 8],
    // "in_channels": 256,
    // "out_channels": 256,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(8, 8), 256, 256, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "averagepool",
    // "input_hw": [8, 8],
    // "in_channels": 256,
    // "pool_hw": [2, 2],
    // "stride": 2
    model.avgPoolingLayer(HW(8, 8), 256, HW(2, 2), HW(2, 2), HW(0, 0));
    // "layer": "relu",
    // "input_dim": 4096
    model.reLULayer(4096);

    // "layer": "cnn",
    // "input_hw": [4, 4],
    // "in_channels": 256,
    // "out_channels": 512,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(4, 4), 256, 512, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "relu",
    // "input_dim": 8192
    model.reLULayer(8192);

    // "layer": "cnn",
    // "input_hw": [4, 4],
    // "in_channels": 512,
    // "out_channels": 512,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(4, 4), 512, 512, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "relu",
    // "input_dim": 8192
    model.reLULayer(8192);

    // "layer": "cnn",
    // "input_hw": [4, 4],
    // "in_channels": 512,
    // "out_channels": 512,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(4, 4), 512, 512, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "averagepool",
    // "input_hw": [4, 4],
    // "in_channels": 512,
    // "pool_hw": [2, 2],
    // "stride": 2
    model.avgPoolingLayer(HW(4, 4), 512, HW(2, 2), HW(2, 2), HW(0, 0));
    // "layer": "relu",
    // "input_dim": 2048
    model.reLULayer(2048);

    // "layer": "cnn",
    // "input_hw": [2, 2],
    // "in_channels": 512,
    // "out_channels": 512,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(2, 2), 512, 512, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "relu",
    // "input_dim": 2048
    model.reLULayer(2048);

    // "layer": "cnn",
    // "input_hw": [2, 2],
    // "in_channels": 512,
    // "out_channels": 512,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(2, 2), 512, 512, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "relu",
    // "input_dim": 2048
    model.reLULayer(2048);

    // "layer": "cnn",
    // "input_hw": [2, 2],
    // "in_channels": 512,
    // "out_channels": 512,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(2, 2), 512, 512, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "averagepool",
    // "input_hw": [2, 2],
    // "in_channels": 512,
    // "pool_hw": [2, 2],
    // "stride": 2
    model.avgPoolingLayer(HW(2, 2), 512, HW(2, 2), HW(2, 2), HW(0, 0));
    // "layer": "relu",
    // "input_dim": 512
    model.reLULayer(512);

    // "layer": "fc",
    // "input_dim": 512,
    // "output_dim": 256
    model.fullyConnectedLayer(512, 256);
    // "layer": "relu",
    // "input_dim": 256
    model.reLULayer(256);

    // "layer": "fc",
    // "input_dim": 256,
    // "output_dim": 256
    model.fullyConnectedLayer(256, 256);
    // "layer": "relu",
    // "input_dim": 256
    model.reLULayer(256);

    // "layer": "fc",
    // "input_dim": 256,
    // "output_dim": 10
    model.fullyConnectedLayer(256, 10);
    // "layer": "relu",
    // "input_dim": 10
    model.reLULayer(10);

    // Forward pass
    stopwatch::timepoint("Start");
    auto output = model.forward(input);
    stopwatch::timepoint("Forward");

    return 0;
}
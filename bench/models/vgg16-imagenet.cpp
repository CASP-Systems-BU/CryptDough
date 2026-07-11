#include <numeric>

#include "cdough.h"

// python3 ../scripts/run_experiment.py -p 3 vgg16-imagenet

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
    const auto inputSize = HW(224, 224);
    const size_t channels_num = 3;
    const size_t precision = 8;

    auto inputPlain = PlainMatrix<DataType>::RandomMatrix(engine, test_size * inputSize.first,
                                                          channels_num * inputSize.second);
    auto input = engine.secret_share_matrix(inputPlain, 0);
    input.setPrecision(precision);

    // VGG16
    cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine> model(engine, precision);

    // "layer": "cnn",
    // "input_hw": [224, 224],
    // "in_channels": 3,
    // "out_channels": 64,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(224, 224), 3, 64, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "relu",
    // "input_dim": 3211264
    model.reLULayer(3211264);

    // "layer": "cnn",
    // "input_hw": [224, 224],
    // "in_channels": 64,
    // "out_channels": 64,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(224, 224), 64, 64, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "averagepool",
    // "input_hw": [224, 224],
    // "in_channels": 64,
    // "pool_hw": [2, 2],
    // "stride": 2
    model.avgPoolingLayer(HW(224, 224), 64, HW(2, 2), HW(2, 2), HW(0, 0));
    // "layer": "relu",
    // "input_dim": 802816
    model.reLULayer(802816);

    // "layer": "cnn",
    // "input_hw": [112, 112],
    // "in_channels": 64,
    // "out_channels": 128,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(112, 112), 64, 128, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "relu",
    // "input_dim": 1605632
    model.reLULayer(1605632);

    // "layer": "cnn",
    // "input_hw": [112, 112],
    // "in_channels": 128,
    // "out_channels": 128,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(112, 112), 128, 128, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "averagepool",
    // "input_hw": [112, 112],
    // "in_channels": 128,
    // "pool_hw": [2, 2],
    // "stride": 2
    model.avgPoolingLayer(HW(112, 112), 128, HW(2, 2), HW(2, 2), HW(0, 0));
    // "layer": "relu",
    // "input_dim": 401408
    model.reLULayer(401408);

    // "layer": "cnn",
    // "input_hw": [56, 56],
    // "in_channels": 128,
    // "out_channels": 256,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(56, 56), 128, 256, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "relu",
    // "input_dim": 802816
    model.reLULayer(802816);

    // "layer": "cnn",
    // "input_hw": [56, 56],
    // "in_channels": 256,
    // "out_channels": 256,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(56, 56), 256, 256, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "relu",
    // "input_dim": 802816
    model.reLULayer(802816);

    // "layer": "cnn",
    // "input_hw": [56, 56],
    // "in_channels": 256,
    // "out_channels": 256,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(56, 56), 256, 256, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "averagepool",
    // "input_hw": [56, 56],
    // "in_channels": 256,
    // "pool_hw": [2, 2],
    // "stride": 2
    model.avgPoolingLayer(HW(56, 56), 256, HW(2, 2), HW(2, 2), HW(0, 0));
    // "layer": "relu",
    // "input_dim": 200704
    model.reLULayer(200704);

    // "layer": "cnn",
    // "input_hw": [28, 28],
    // "in_channels": 256,
    // "out_channels": 512,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(28, 28), 256, 512, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "relu",
    // "input_dim": 401408
    model.reLULayer(401408);

    // "layer": "cnn",
    // "input_hw": [28, 28],
    // "in_channels": 512,
    // "out_channels": 512,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(28, 28), 512, 512, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "relu",
    // "input_dim": 401408
    model.reLULayer(401408);

    // "layer": "cnn",
    // "input_hw": [28, 28],
    // "in_channels": 512,
    // "out_channels": 512,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(28, 28), 512, 512, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "averagepool",
    // "input_hw": [28, 28],
    // "in_channels": 512,
    // "pool_hw": [2, 2],
    // "stride": 2
    model.avgPoolingLayer(HW(28, 28), 512, HW(2, 2), HW(2, 2), HW(0, 0));
    // "layer": "relu",
    // "input_dim": 100352
    model.reLULayer(100352);

    // "layer": "cnn",
    // "input_hw": [14, 14],
    // "in_channels": 512,
    // "out_channels": 512,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(14, 14), 512, 512, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "relu",
    // "input_dim": 100352
    model.reLULayer(100352);

    // "layer": "cnn",
    // "input_hw": [14, 14],
    // "in_channels": 512,
    // "out_channels": 512,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(14, 14), 512, 512, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "relu",
    // "input_dim": 100352
    model.reLULayer(100352);

    // "layer": "cnn",
    // "input_hw": [2, 2],
    // "in_channels": 512,
    // "out_channels": 512,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(14, 14), 512, 512, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "averagepool",
    // "input_hw": [2, 2],
    // "in_channels": 512,
    // "pool_hw": [2, 2],
    // "stride": 2
    model.avgPoolingLayer(HW(14, 14), 512, HW(2, 2), HW(2, 2), HW(0, 0));
    // "layer": "relu",
    // "input_dim": 25088
    model.reLULayer(25088);

    // Starting here: the fully connected layers tailored for ImageNet classification (1000 classes)

    // "layer": "averagepool",
    // "input_hw": [2, 2],
    // "in_channels": 512,
    // "pool_hw": [2, 2],
    // "stride": 2
    model.avgPoolingLayer(HW(7, 7), 512, HW(2, 2), HW(2, 2), HW(0, 0));
    // "layer": "relu",
    // "input_dim": 4608
    model.reLULayer(4608);

    // "layer": "fc",
    // "input_dim": 4608,
    // "output_dim": 4096
    model.fullyConnectedLayer(4608, 4096);
    // "layer": "relu",
    // "input_dim": 4096
    model.reLULayer(4096);

    // "layer": "fc",
    // "input_dim": 4096,
    // "output_dim": 4096
    model.fullyConnectedLayer(4096, 4096);
    // "layer": "relu",
    // "input_dim": 4096
    model.reLULayer(4096);

    // "layer": "fc",
    // "input_dim": 4096,
    // "output_dim": 1000
    model.fullyConnectedLayer(4096, 1000);
    // "layer": "relu",
    // "input_dim": 1000
    model.reLULayer(1000);

    // Forward pass
    stopwatch::timepoint("Start");
    auto output = model.forward(input);
    stopwatch::timepoint("Forward");

    return 0;
}
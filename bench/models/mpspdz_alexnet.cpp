#include <numeric>

#include "cdough.h"

// MP-SPDZ AlexNet (AvgPool) on CIFAR-10
// python3 ../scripts/run_experiment.py -p 3 -r 16 mpspdz_alexnet

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
        std::cout << "Using test size: " << test_size << " and batch size: " << engine.getBatchSize() << std::endl;

    // test data
    // CIFAR-10: 32x32x3
    const auto inputSize = HW(32, 32);
    const size_t channels_num = 3;
    const size_t precision = 16;

    auto inputPlain = PlainMatrix<DataType>::RandomMatrix(engine, test_size * inputSize.first, channels_num * inputSize.second);
    auto input = engine.secret_share_matrix(inputPlain, 0);
    input.setPrecision(precision);

    // MP-SPDZ AlexNet (AvgPool version)
    // layer order follows MP-SPDZ: conv, ReLU, avgpool
    cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine> model(engine, precision);
    // conv, ReLU, and AvgPool
    // Conv2d(3, 64, kernel_size=3, stride=1, padding=2)
    // input: 32×32×3 → output: 34×34×64
    model.conv2DLayer(HW(32, 32), 3, 64, HW(3, 3), HW(1, 1), HW(2, 2));
    // ReLU: 34×34×64 = 73984
    model.reLULayer(73984);
    // AvgPool2d(kernel_size=2, stride=2)
    // input: 34×34×64 
    // output: 17×17×64
    model.avgPoolingLayer(HW(34, 34), 64, HW(2, 2), HW(2, 2), HW(0, 0));

    // Conv2d(64, 96, kernel_size=3, stride=1, padding=2)
    // input: 17×17×64
    // output: 19×19×96
    model.conv2DLayer(HW(17, 17), 64, 96, HW(3, 3), HW(1, 1), HW(2, 2));
    // ReLU: 19×19×96 = 34656
    model.reLULayer(34656);
    // AvgPool2d(kernel_size=2, stride=2)
    // input: 19×19×96 
    // output: 9×9×96
    model.avgPoolingLayer(HW(19, 19), 96, HW(2, 2), HW(2, 2), HW(0, 0));

    // conv and ReLU (no pool)
    // Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
    // input: 9×9×96
    // output: 9×9×96
    model.conv2DLayer(HW(9, 9), 96, 96, HW(3, 3), HW(1, 1), HW(1, 1));
    // ReLU: 9×9×96 = 7776
    model.reLULayer(7776);

    // Conv2d(96, 64, kernel_size=3, stride=1, padding=1)
    // input: 9×9×96 
    // output: 9×9×64
    model.conv2DLayer(HW(9, 9), 96, 64, HW(3, 3), HW(1, 1), HW(1, 1));
    // ReLU: 9×9×64 = 5184
    model.reLULayer(5184);

    // conv, ReLU, and AvgPool
    // Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    // input: 9×9×64 
    // output: 9×9×64
    model.conv2DLayer(HW(9, 9), 64, 64, HW(3, 3), HW(1, 1), HW(1, 1));
    // ReLU: 9×9×64 = 5184
    model.reLULayer(5184);
    // AvgPool2d(kernel_size=3, stride=2)
    // input: 9×9×64
    // output: 4×4×64
    model.avgPoolingLayer(HW(9, 9), 64, HW(3, 3), HW(2, 2), HW(0, 0));

    // flatten: 4×4×64 = 1024

    // FC
    model.fullyConnectedLayer(1024, 128);
    model.reLULayer(128);

    model.fullyConnectedLayer(128, 256);
    model.reLULayer(256);

    model.fullyConnectedLayer(256, 10);

    // forward pass
    stopwatch::timepoint("Start");
    auto output = model.forward(input);
    stopwatch::timepoint("Forward");

    return 0;
}
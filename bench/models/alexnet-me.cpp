#include <mutex>
#include <numeric>

#include "cdough.h"

// python3 ../scripts/run_experiment.py -p 3 alexnet_me

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;

using DataType = int64_t;
using HW = cdough::matrix::HeightWidth;

template <typename Engine>
void run_alexnet_test(Engine& engine, size_t test_size) {
    // Test Data
    const auto inputSize = HW(32, 32);
    const size_t channels_num = 3;
    const size_t precision = 8;

    auto inputPlain = PlainMatrix<DataType>::RandomMatrix(engine, test_size * inputSize.first,
                                                          channels_num * inputSize.second);
    auto input = engine.secret_share_matrix(inputPlain, 0);
    input.setPrecision(precision);

    // AlexNet
    cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine> model(engine, precision);

    // "layer": "cnn",
    // "input_hw": [32, 32],
    // "in_channels": 3,
    // "out_channels": 96,
    // "filter_hw": [11, 11],
    // "stride": 4,
    // "padding": 9
    model.conv2DLayer(HW(32, 32), 3, 96, HW(11, 11), HW(4, 4), HW(9, 9));
    // "layer": "averagepool",
    // "input_hw": [10, 10],
    // "in_channels": 96,
    // "pool_hw": [3, 3],
    // "stride": 2
    model.avgPoolingLayer(HW(10, 10), 96, HW(3, 3), HW(2, 2), HW(0, 0));
    // "layer": "relu",
    // "input_dim": 1536
    model.reLULayer(1536);

    // "layer": "cnn",
    // "input_hw": [4, 4],
    // "in_channels": 96,
    // "out_channels": 256,
    // "filter_hw": [5, 5],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(4, 4), 96, 256, HW(5, 5), HW(1, 1), HW(1, 1));
    // "layer": "averagepool",
    // "input_hw": [2, 2],
    // "in_channels": 256,
    // "pool_hw": [2, 2],
    // "stride": 1
    model.avgPoolingLayer(HW(2, 2), 256, HW(2, 2), HW(1, 1), HW(0, 0));
    // "layer": "relu",
    // "input_dim": 256
    model.reLULayer(256);

    // "layer": "cnn",
    // "input_hw": [1, 1],
    // "in_channels": 256,
    // "out_channels": 384,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(1, 1), 256, 384, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "relu",
    // "input_dim": 384
    model.reLULayer(384);

    // "layer": "cnn",
    // "input_hw": [1, 1],
    // "in_channels": 384,
    // "out_channels": 384,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(1, 1), 384, 384, HW(3, 3), HW(1, 1), HW(1, 1));
    // "layer": "relu",
    // "input_dim": 384
    model.reLULayer(384);

    // "layer": "cnn",
    // "input_hw": [1, 1],
    // "in_channels": 384,
    // "out_channels": 256,
    // "filter_hw": [3, 3],
    // "stride": 1,
    // "padding": 1
    model.conv2DLayer(HW(1, 1), 384, 256, HW(3, 3), HW(1, 1), HW(1, 1));
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
    auto output = model.forward(input);
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();
    auto test_size = engine.getArg<size_t>("test-size", "r", 192);
    auto processes_num = engine.getArg<size_t>("num-engines", "neng", 1);

    if (pID == 0)
        std::cout << "Using test size: " << test_size
                  << " and batch size: " << engine.getBatchSize() << std::endl;

    stopwatch::timepoint("Start");

    // create different threads, each doing a alexnet test inside a lambda function.
    assert(test_size % processes_num == 0);  // ensure test size is divisible by number of engines
    std::vector<std::thread> threads;
    std::mutex cdough_init_mutex;
    for (int i = 0; i < processes_num; i++) {
        threads.emplace_back([&, i]() {
            cdough_init_mutex.lock();
            EngineRef eng = cdough_init(argc, argv);
            cdough_init_mutex.unlock();
            run_alexnet_test(eng, test_size / processes_num);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
    stopwatch::timepoint("Forward");

    return 0;
}

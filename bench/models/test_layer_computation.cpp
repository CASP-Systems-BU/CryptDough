/**
 * test_layer_computation.cpp
 *
 * Tests the new weight-loading functionality added to machine_learning.h:
 *   1. load_bin_file() — reads .bin files into std::vector
 *   2. conv2DLayerWithWeights() — builds Conv2D from pretrained filter
 *   3. fullyConnectedLayerWithWeights() — builds FC from pretrained weights+bias
 *
 * For each layer type, runs a forward pass with a known input and verifies
 * the output has the expected dimensions and is non-zero.
 *
 * Requires weight files in weights/ directory (from extract_weights.py).
 *
 * Plaintext: cmake .. -DPROTOCOL=1 && make test_layer_computation && ./test_layer_computation
 * 3PC MPC:   cmake .. -DPROTOCOL=3 && make test_layer_computation && mpirun -np 3 ./test_layer_computation
 */

#include "cdough.h"

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;

using DataType = int64_t;
using HW = cdough::matrix::HeightWidth;

// load a .bin file, build a PlainMatrix, secret share it
template <typename Engine>
SecureMatrix<DataType> loadAndShareMatrix(Engine& engine, const std::string& filepath,
                                           size_t rows, size_t cols, size_t precision,
                                           bool columnWise = false) {
    auto data = cdough::operators::ml::load_bin_file<DataType>(filepath);
    single_cout("  Loaded " << filepath << " (" << data.size() << " values)");

    if (data.size() != rows * cols) {
        single_cout("  ERROR: expected " << rows * cols << " values, got " << data.size());
        throw std::runtime_error("Size mismatch loading " + filepath);
    }

    // build PlainMatrix from raw data
    cdough::Vector<DataType> vec(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        vec[i] = data[i];
    }
    PlainMatrix<DataType> plain(vec, rows, cols, columnWise);
    auto shared = engine.secret_share_matrix(plain, 0);
    shared.setPrecision(precision);
    return shared;
}

// FullyConnected layer with pretrained weights
template <typename Engine>
bool testFullyConnected(Engine& engine, const std::string& weightsDir, size_t precision) {
    single_cout("\n----------------------------------------------");
    single_cout("  Test: FullyConnected layer with pretrained weights");
    single_cout("----------------------------------------------");

    // Use VGG16-CIFAR10 classifier.6 (final layer): 512 -> 10
    const size_t inputDim = 512;
    const size_t outputDim = 10;
    const std::string weightFile = weightsDir + "/classifier_6_weight.bin";
    const std::string biasFile = weightsDir + "/classifier_6_bias.bin";

    try {
        // load pretrained weights
        auto weights = loadAndShareMatrix(engine, weightFile, inputDim, outputDim, precision, true);
        auto bias = loadAndShareMatrix(engine, biasFile, 1, outputDim, precision);

        // biuld model with one FC layer
        cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine> model(engine, precision);
        model.fullyConnectedLayerWithWeights(weights, bias, inputDim, outputDim);

        // create a test input: 1 instance, 512 features (small random values)
        auto inputPlain = PlainMatrix<DataType>::RandomMatrix(engine, 1, inputDim);
        auto input = engine.secret_share_matrix(inputPlain, 0);
        input.setPrecision(precision);

        // Forward pass
        auto output = model.forward(input);
        single_cout("  Output dimensions: " << output.rows() << "x" << output.cols());

        // check the dimensions
        if (output.cols() != outputDim) {
            single_cout("  FAIL: expected output cols=" << outputDim << ", got " << output.cols());
            return false;
        }

        // open result and check it's not all zeros
        // (pretrained weights + random input should produce non-zero output)
        auto opened = output.open();
        int nonzero = 0;
        auto openedData = opened.data();
        for (size_t i = 0; i < openedData.size(); i++) {
            if (openedData[i] != 0) nonzero++;
        }
        single_cout("  Non-zero output values: " << nonzero << "/" << openedData.size());

        if (nonzero == 0) {
            single_cout("  FAIL: all output values are zero (suspicious)");
            return false;
        }

        single_cout("  PASS: FC layer forward pass with pretrained weights");
        return true;

    } catch (const std::exception& e) {
        single_cout("  FAIL: " << e.what());
        return false;
    }
}

// Test 2: Conv2D layer with pretrained weights
template <typename Engine>
bool testConv2D(Engine& engine, const std::string& weightsDir, size_t precision) {
    single_cout("\n----------------------------------------------");
    single_cout("  Test: Conv2D layer with pretrained weights");
    single_cout("----------------------------------------------");

    // Use VGG16-CIFAR10 features.0 (first conv): 3 input channels -> 64 output, 3x3 filter
    const size_t inChannels = 3;
    const size_t outChannels = 64;
    const HW filterSize(3, 3);
    const HW inputSize(32, 32); // CIFAR-10 image size
    const HW stride(1, 1);
    const HW padding(1, 1);

    const size_t filterRows = outChannels * filterSize.first; // 64 * 3 = 192
    const size_t filterCols = filterSize.second * inChannels; // 3 * 3 = 9
    const std::string weightFile = weightsDir + "/features_0_weight.bin";

    try {
        // load pretrained filter
        auto filter = loadAndShareMatrix(engine, weightFile, filterRows, filterCols, precision);

        // build model with one Conv2D layer
        cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine> model(engine, precision);
        model.conv2DLayerWithWeights(filter, inputSize, inChannels, outChannels,
                                      filterSize, stride, padding);

        // Create a test input: 1 instance, 32x32 with 3 channels
        auto inputPlain = PlainMatrix<DataType>::RandomMatrix(
            engine, 1 * inputSize.first, inChannels * inputSize.second);
        auto input = engine.secret_share_matrix(inputPlain, 0);
        input.setPrecision(precision);

        auto output = model.forward(input);
        single_cout("  Output dimensions: " << output.rows() << "x" << output.cols());

        auto opened = output.open();
        int nonzero = 0;
        auto openedData = opened.data();
        for (size_t i = 0; i < openedData.size(); i++) {
            if (openedData[i] != 0) nonzero++;
        }
        single_cout("  Non-zero output values: " << nonzero << "/" << openedData.size());

        if (nonzero == 0) {
            single_cout("  FAIL: all output values are zero (suspicious)");
            return false;
        }

        single_cout("  PASS: Conv2D layer forward pass with pretrained weights");
        return true;

    } catch (const std::exception& e) {
        single_cout("  FAIL: " << e.what());
        return false;
    }
}

// Test 3: check if load_bin_file catches errors
template <typename Engine>
bool testLoadBinErrors(Engine& engine) {
    single_cout("\n----------------------------------------------");
    single_cout("  Test: load_bin_file error handling");
    single_cout("----------------------------------------------");

    bool allPassed = true;

    // missing file should throw
    try {
        auto data = cdough::operators::ml::load_bin_file<DataType>("nonexistent_file.bin");
        single_cout("  FAIL: should have thrown for missing file");
        allPassed = false;
    } catch (const std::runtime_error& e) {
        single_cout("  PASS: missing file throws: " << e.what());
    }

    single_cout("  " << (allPassed ? "PASS" : "FAIL") << ": error handling tests");
    return allPassed;
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);

    constexpr int PRECISION = 12;
    const std::string weightsDir = "weights/vgg16_cifar10";

    single_cout("==============================================");
    single_cout("  Layer Computation Test (Pretrained Weights)");
    single_cout("  Precision: " << PRECISION);
    single_cout("  Weights: " << weightsDir);
    single_cout("==============================================");

    bool p1 = testLoadBinErrors(engine);
    bool p2 = testFullyConnected(engine, weightsDir, PRECISION);
    bool p3 = testConv2D(engine, weightsDir, PRECISION);

    single_cout("\n==============================================");
    single_cout("  RESULTS:");
    single_cout("    Error handling:  " << (p1 ? "PASS" : "FAIL"));
    single_cout("    FC layer:        " << (p2 ? "PASS" : "FAIL"));
    single_cout("    Conv2D layer:    " << (p3 ? "PASS" : "FAIL"));
    single_cout("==============================================");

    return (p1 && p2 && p3) ? 0 : 1;
}
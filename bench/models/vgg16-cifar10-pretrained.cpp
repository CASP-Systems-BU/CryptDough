/**
 * vgg16-cifar10-pretrained.cpp
 *
 * VGG16-BN (CIFAR-10) pretrained model architecture for cdough.
 * Loads real weights from chenyaofo/pytorch-cifar-models (94.16% acc).
 *
 * Architecture (from PyTorch):
 *   13 Conv2d layers (3x3, stride 1, padding 1)
 *   13 BatchNorm2d layers (skipped — not yet implemented in cdough)
 *   13 ReLU activations
 *   5 MaxPool2d (approximated with AvgPool 2x2, stride 2)
 *   3 FC layers: 512->512, 512->512, 512->10
 *
 * Weight files: weights/vgg16_cifar10/ (from extract_weights.py)
 *
 * Plaintext:
 *   cmake .. -DPROTOCOL=1 && make vgg16-cifar10-pretrained
 *   ./vgg16-cifar10-pretrained
 *
 * 3PC MPC:
 *   cmake .. -DPROTOCOL=3 && make vgg16-cifar10-pretrained
 *   mpirun -np 3 ./vgg16-cifar10-pretrained
 */

#include "cdough.h"

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;

using DataType = int64_t;
using HW = cdough::matrix::HeightWidth;

// Helper: load a .bin file into a PlainMatrix and secret share it
template <typename Engine>
SecureMatrix<DataType> loadAndShare(Engine& engine, const std::string& filepath,
                                    size_t rows, size_t cols, size_t precision,
                                    bool columnWise = false) {
    auto data = cdough::operators::ml::load_bin_file<DataType>(filepath);
    if (data.size() != rows * cols) {
        throw std::runtime_error("Size mismatch loading " + filepath +
            ": expected " + std::to_string(rows * cols) +
            ", got " + std::to_string(data.size()));
    }
    cdough::Vector<DataType> vec(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        vec[i] = data[i];
    }
    PlainMatrix<DataType> plain(vec, rows, cols, columnWise);
    auto shared = engine.secret_share_matrix(plain, 0);
    shared.setPrecision(precision);
    return shared;
}

// Helper: add a Conv2D layer with pretrained weights
template <typename Engine>
void addConv(Engine& engine,
             cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine>& model,
             const std::string& weightsDir, const std::string& layerIdx,
             const HW& inputSize, size_t inChannels, size_t outChannels,
             const HW& filterSize, const HW& stride, const HW& padding,
             size_t precision) {
    size_t filterRows = outChannels * filterSize.first;
    size_t filterCols = filterSize.second * inChannels;
    std::string filterPath = weightsDir + "/features_" + layerIdx + "_weight.bin";

    single_cout("  Loading Conv2D features." << layerIdx
                << " (" << inChannels << "->" << outChannels << ")");

    auto filter = loadAndShare(engine, filterPath, filterRows, filterCols, precision);
    model.conv2DLayerWithWeights(filter, inputSize, inChannels, outChannels,
                                  filterSize, stride, padding);
}

// Helper: add an FC layer with pretrained weights
template <typename Engine>
void addFC(Engine& engine,
           cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine>& model,
           const std::string& weightsDir, const std::string& layerIdx,
           size_t inputDim, size_t outputDim, size_t precision) {
    std::string weightPath = weightsDir + "/classifier_" + layerIdx + "_weight.bin";
    std::string biasPath = weightsDir + "/classifier_" + layerIdx + "_bias.bin";

    single_cout("  Loading FC classifier." << layerIdx
                << " (" << inputDim << "->" << outputDim << ")");

    auto weights = loadAndShare(engine, weightPath, inputDim, outputDim, precision, true);
    auto bias = loadAndShare(engine, biasPath, 1, outputDim, precision);
    model.fullyConnectedLayerWithWeights(weights, bias, inputDim, outputDim);
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);

    const size_t precision = 12;
    const std::string weightsDir = "weights/vgg16_cifar10";

    // Batch size: configurable via --test-size / -r flag (default 1)
    auto batchSize = engine.getArg<size_t>("test-size", "r", 1);

    single_cout("==============================================");
    single_cout("  VGG16-BN (CIFAR-10) Pretrained Inference");
    single_cout("  Precision: " << precision);
    single_cout("  Batch size: " << batchSize);
    single_cout("  Weights: " << weightsDir);
    single_cout("==============================================");

    // Build the model
    cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine> model(engine, precision);

    single_cout("\n  Building model...");

    // ===== Block 1: 2 conv layers, pool =====
    // features.0: Conv2d(3, 64, 3x3, padding=1)    input: 32x32
    // features.1: BatchNorm2d(64)                    (skipped)
    // features.2: ReLU
    // features.3: Conv2d(64, 64, 3x3, padding=1)   input: 32x32
    // features.4: BatchNorm2d(64)                    (skipped)
    // features.5: ReLU
    // features.6: MaxPool2d(2,2)                     -> 16x16
    addConv(engine, model, weightsDir, "0",  HW(32,32), 3,  64, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(64 * 32 * 32);
    addConv(engine, model, weightsDir, "3",  HW(32,32), 64, 64, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(64 * 32 * 32);
    model.avgPoolingLayer(HW(32,32), 64, HW(2,2), HW(2,2), HW(0,0));

    // ===== Block 2: 2 conv layers, pool =====
    // features.7:  Conv2d(64, 128, 3x3, padding=1)  input: 16x16
    // features.8:  BatchNorm2d(128)                   (skipped)
    // features.9:  ReLU
    // features.10: Conv2d(128, 128, 3x3, padding=1) input: 16x16
    // features.11: BatchNorm2d(128)                   (skipped)
    // features.12: ReLU
    // features.13: MaxPool2d(2,2)                     -> 8x8
    addConv(engine, model, weightsDir, "7",  HW(16,16), 64,  128, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(128 * 16 * 16);
    addConv(engine, model, weightsDir, "10", HW(16,16), 128, 128, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(128 * 16 * 16);
    model.avgPoolingLayer(HW(16,16), 128, HW(2,2), HW(2,2), HW(0,0));

    // ===== Block 3: 3 conv layers, pool =====
    // features.14: Conv2d(128, 256, 3x3, padding=1) input: 8x8
    // features.15: BatchNorm2d(256)                   (skipped)
    // features.16: ReLU
    // features.17: Conv2d(256, 256, 3x3, padding=1) input: 8x8
    // features.18: BatchNorm2d(256)                   (skipped)
    // features.19: ReLU
    // features.20: Conv2d(256, 256, 3x3, padding=1) input: 8x8
    // features.21: BatchNorm2d(256)                   (skipped)
    // features.22: ReLU
    // features.23: MaxPool2d(2,2)                     -> 4x4
    addConv(engine, model, weightsDir, "14", HW(8,8), 128, 256, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(256 * 8 * 8);
    addConv(engine, model, weightsDir, "17", HW(8,8), 256, 256, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(256 * 8 * 8);
    addConv(engine, model, weightsDir, "20", HW(8,8), 256, 256, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(256 * 8 * 8);
    model.avgPoolingLayer(HW(8,8), 256, HW(2,2), HW(2,2), HW(0,0));

    // ===== Block 4: 3 conv layers, pool =====
    // features.24: Conv2d(256, 512, 3x3, padding=1) input: 4x4
    // features.25: BatchNorm2d(512)                   (skipped)
    // features.26: ReLU
    // features.27: Conv2d(512, 512, 3x3, padding=1) input: 4x4
    // features.28: BatchNorm2d(512)                   (skipped)
    // features.29: ReLU
    // features.30: Conv2d(512, 512, 3x3, padding=1) input: 4x4
    // features.31: BatchNorm2d(512)                   (skipped)
    // features.32: ReLU
    // features.33: MaxPool2d(2,2)                     -> 2x2
    addConv(engine, model, weightsDir, "24", HW(4,4), 256, 512, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(512 * 4 * 4);
    addConv(engine, model, weightsDir, "27", HW(4,4), 512, 512, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(512 * 4 * 4);
    addConv(engine, model, weightsDir, "30", HW(4,4), 512, 512, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(512 * 4 * 4);
    model.avgPoolingLayer(HW(4,4), 512, HW(2,2), HW(2,2), HW(0,0));

    // ===== Block 5: 3 conv layers, pool =====
    // features.34: Conv2d(512, 512, 3x3, padding=1) input: 2x2
    // features.35: BatchNorm2d(512)                   (skipped)
    // features.36: ReLU
    // features.37: Conv2d(512, 512, 3x3, padding=1) input: 2x2
    // features.38: BatchNorm2d(512)                   (skipped)
    // features.39: ReLU
    // features.40: Conv2d(512, 512, 3x3, padding=1) input: 2x2
    // features.41: BatchNorm2d(512)                   (skipped)
    // features.42: ReLU
    // features.43: MaxPool2d(2,2)                     -> 1x1
    addConv(engine, model, weightsDir, "34", HW(2,2), 512, 512, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(512 * 2 * 2);
    addConv(engine, model, weightsDir, "37", HW(2,2), 512, 512, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(512 * 2 * 2);
    addConv(engine, model, weightsDir, "40", HW(2,2), 512, 512, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(512 * 2 * 2);
    model.avgPoolingLayer(HW(2,2), 512, HW(2,2), HW(2,2), HW(0,0));

    // ===== Classifier =====
    // classifier.0: Linear(512, 512)
    // classifier.1: ReLU
    // classifier.2: Dropout (skipped)
    // classifier.3: Linear(512, 512)
    // classifier.4: ReLU
    // classifier.5: Dropout (skipped)
    // classifier.6: Linear(512, 10)
    addFC(engine, model, weightsDir, "0", 512, 512, precision);
    model.reLULayer(512);
    addFC(engine, model, weightsDir, "3", 512, 512, precision);
    model.reLULayer(512);
    addFC(engine, model, weightsDir, "6", 512, 10, precision);

    single_cout("  Model built: 13 Conv2D + 13 ReLU + 5 AvgPool + 3 FC + 2 ReLU");

    // Create test input: CIFAR-10 image (32x32, 3 channels)
    single_cout("\n  Running inference...");
    auto inputPlain = PlainMatrix<DataType>::RandomMatrix(
        engine, batchSize * 32, 3 * 32);
    auto input = engine.secret_share_matrix(inputPlain, 0);
    input.setPrecision(precision);

    // Forward pass
    auto output = model.forward(input);
    auto opened = output.open();

    single_cout("\n  Output: " << opened.rows() << "x" << opened.cols());
    auto outData = opened.data();

    // Print predictions for each instance in the batch
    for (size_t b = 0; b < batchSize; b++) {
        single_cout("\n  Instance " << b << " logits:");
        DataType maxVal = outData[b * 10];
        size_t maxIdx = 0;
        for (size_t c = 0; c < 10; c++) {
            DataType val = outData[b * 10 + c];
            single_cout("    class " << c << ": " << val);
            if (val > maxVal) {
                maxVal = val;
                maxIdx = c;
            }
        }
        single_cout("  Predicted class: " << maxIdx);
    }

    single_cout("\n==============================================");
    single_cout("  Done.");
    single_cout("==============================================");

    return 0;
}
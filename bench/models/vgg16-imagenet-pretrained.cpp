/**
 * vgg16-imagenet-pretrained.cpp
 *
 * VGG16 (ImageNet) pretrained model for cdough.
 * Loads weights from torchvision.models.vgg16(pretrained=True).
 *
 * Architecture:
 *   13 Conv2d layers (3x3, stride 1, padding 1) — bias NOT loaded
 *   13 ReLU activations
 *   5 MaxPool2d (approximated with AvgPool 2x2, stride 2)
 *   3 FC layers: 25088->4096, 4096->4096, 4096->1000
 *
 * Note: VGG16-ImageNet has no BatchNorm. Conv biases exist but
 * conv2DLayerWithWeights does not accept a bias parameter, so they
 * are skipped. This will affect output accuracy.
 *
 * Note: Conv2D padding=1 is currently broken in cdough (see isolation
 * tests). Predictions will not be meaningful until padding is fixed.
 *
 * Weight files: weights/vgg16_imagenet/ (from extract_weights.py)
 * Instance files: instances/imagenet/ (from extract_weights.py --instances)
 *
 * Usage:
 *   cmake .. -DPROTOCOL=3 && make vgg16-imagenet-pretrained
 *   mpirun -np 3 ./vgg16-imagenet-pretrained
 */

#include "cdough.h"

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;

using DataType = int64_t;
using HW = cdough::matrix::HeightWidth;

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
    for (size_t i = 0; i < data.size(); i++) vec[i] = data[i];
    PlainMatrix<DataType> plain(vec, rows, cols, columnWise);
    auto shared = engine.secret_share_matrix(plain, 0);
    shared.setPrecision(precision);
    return shared;
}

template <typename Engine>
void addConv(Engine& engine,
             cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine>& model,
             const std::string& dir, const std::string& idx,
             const HW& inputSize, size_t inCh, size_t outCh,
             const HW& filt, const HW& stride, const HW& pad,
             size_t precision) {
    size_t rows = outCh * filt.first;
    size_t cols = filt.second * inCh;
    auto filter = loadAndShare(engine, dir + "/features_" + idx + "_weight.bin",
                               rows, cols, precision);
    single_cout("  Conv features." << idx << " (" << inCh << "->" << outCh << ")");
    model.conv2DLayerWithWeights(filter, inputSize, inCh, outCh, filt, stride, pad);
}

template <typename Engine>
void addFC(Engine& engine,
           cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine>& model,
           const std::string& dir, const std::string& idx,
           size_t inDim, size_t outDim, size_t precision) {
    auto w = loadAndShare(engine, dir + "/classifier_" + idx + "_weight.bin",
                          inDim, outDim, precision, true);
    auto b = loadAndShare(engine, dir + "/classifier_" + idx + "_bias.bin",
                          1, outDim, precision);
    single_cout("  FC classifier." << idx << " (" << inDim << "->" << outDim << ")");
    model.fullyConnectedLayerWithWeights(w, b, inDim, outDim);
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);

    const size_t precision = 12;
    const std::string weightsDir = "weights/vgg16_imagenet";
    const std::string instanceDir = "instances/imagenet";

    auto batchSize = engine.getArg<size_t>("test-size", "r", 1);
    auto instanceName = engine.getArg<std::string>("instance", "i", "labrador");

    single_cout("VGG16 (ImageNet) Pretrained Inference");
    single_cout("  Precision: " << precision);
    single_cout("  Batch size: " << batchSize);
    single_cout("  Instance: " << instanceName);

    // build model
    cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine> model(engine, precision);

    // VGG16 ImageNet: no BatchNorm, layer indices differ from VGG16-BN
    //
    // Block 1: features.0, features.2                    224x224 -> pool -> 112x112
    // Block 2: features.5, features.7                    112x112 -> pool -> 56x56
    // Block 3: features.10, features.12, features.14     56x56   -> pool -> 28x28
    // Block 4: features.17, features.19, features.21     28x28   -> pool -> 14x14
    // Block 5: features.24, features.26, features.28     14x14   -> pool -> 7x7

    // block 1
    addConv(engine, model, weightsDir, "0",  HW(224,224), 3,  64, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(64 * 224 * 224);
    addConv(engine, model, weightsDir, "2",  HW(224,224), 64, 64, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(64 * 224 * 224);
    model.avgPoolingLayer(HW(224,224), 64, HW(2,2), HW(2,2), HW(0,0));

    // block 2
    addConv(engine, model, weightsDir, "5",  HW(112,112), 64,  128, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(128 * 112 * 112);
    addConv(engine, model, weightsDir, "7",  HW(112,112), 128, 128, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(128 * 112 * 112);
    model.avgPoolingLayer(HW(112,112), 128, HW(2,2), HW(2,2), HW(0,0));

    // block 3
    addConv(engine, model, weightsDir, "10", HW(56,56), 128, 256, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(256 * 56 * 56);
    addConv(engine, model, weightsDir, "12", HW(56,56), 256, 256, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(256 * 56 * 56);
    addConv(engine, model, weightsDir, "14", HW(56,56), 256, 256, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(256 * 56 * 56);
    model.avgPoolingLayer(HW(56,56), 256, HW(2,2), HW(2,2), HW(0,0));

    // block 4
    addConv(engine, model, weightsDir, "17", HW(28,28), 256, 512, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(512 * 28 * 28);
    addConv(engine, model, weightsDir, "19", HW(28,28), 512, 512, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(512 * 28 * 28);
    addConv(engine, model, weightsDir, "21", HW(28,28), 512, 512, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(512 * 28 * 28);
    model.avgPoolingLayer(HW(28,28), 512, HW(2,2), HW(2,2), HW(0,0));

    // block 5
    addConv(engine, model, weightsDir, "24", HW(14,14), 512, 512, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(512 * 14 * 14);
    addConv(engine, model, weightsDir, "26", HW(14,14), 512, 512, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(512 * 14 * 14);
    addConv(engine, model, weightsDir, "28", HW(14,14), 512, 512, HW(3,3), HW(1,1), HW(1,1), precision);
    model.reLULayer(512 * 14 * 14);
    model.avgPoolingLayer(HW(14,14), 512, HW(2,2), HW(2,2), HW(0,0));

    // classifier: 512*7*7 = 25088 -> 4096 -> 4096 -> 1000
    addFC(engine, model, weightsDir, "0", 25088, 4096, precision);
    model.reLULayer(4096);
    addFC(engine, model, weightsDir, "3", 4096, 4096, precision);
    model.reLULayer(4096);
    addFC(engine, model, weightsDir, "6", 4096, 1000, precision);

    single_cout("  Model built: 13 Conv2D + 13 ReLU + 5 AvgPool + 3 FC + 2 ReLU");

    // load instance
    std::string instancePath = instanceDir + "/" + instanceName + ".bin";
    single_cout("\n  Loading instance: " << instancePath);

    auto input = loadAndShare(engine, instancePath, batchSize * 224, 3 * 224, precision);

    // forward pass
    single_cout("  Running inference...");
    auto output = model.forward(input);
    auto opened = output.open();

    single_cout("  Output: " << opened.rows() << "x" << opened.cols());
    auto outData = opened.data();

    // print top predictions
    for (size_t b = 0; b < batchSize; b++) {
        size_t offset = b * 1000;

        // find top-5
        std::vector<std::pair<DataType, size_t>> scores(1000);
        for (size_t c = 0; c < 1000; c++) {
            scores[c] = {outData[offset + c], c};
        }
        std::sort(scores.begin(), scores.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        single_cout("\n  Instance " << b << " top-5 predictions:");
        for (int k = 0; k < 5; k++) {
            single_cout("    " << k+1 << ". class " << scores[k].second
                        << " (logit: " << scores[k].first << ")");
        }
    }

    single_cout("\nDone.");
    return 0;
}
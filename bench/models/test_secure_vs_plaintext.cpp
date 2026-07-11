/**
 * test_secure_vs_plaintext.cpp
 *
 * Per-layer correctness test: runs the SAME computation through the SAME
 * ModelML forward pass, saves opened results to a file. User runs this under
 * PROTOCOL=1 (plaintext) and PROTOCOL=3 (3PC MPC), then compares outputs.
 *
 * Usage:
 *   # Step 1: Run in plaintext mode
 *   cmake .. -DPROTOCOL=1 && make test_secure_vs_plaintext
 *   ./test_secure_vs_plaintext
 *   # → saves results to layer_results_protocol1.txt
 *
 *   # Step 2: Run in 3PC mode
 *   cmake .. -DPROTOCOL=3 && make test_secure_vs_plaintext
 *   mpirun -np 3 ./test_secure_vs_plaintext
 *   # → saves results to layer_results_protocol3.txt
 *
 *   # Step 3: Compare (done automatically if both files exist)
 *
 * Uses deterministic input (seeded from weight data) so both runs produce
 * comparable results.
 */

#include "cdough.h"
#include <fstream>
#include <sstream>
#include <cmath>

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;

using DataType = int64_t;
using HW = cdough::matrix::HeightWidth;

// load .bin file into a PlainMatrix
PlainMatrix<DataType> loadPlainMatrix(EngineRef& engine, const std::string& filepath,
                                       size_t rows, size_t cols, bool columnWise = false) {
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
    return PlainMatrix<DataType>(vec, rows, cols, columnWise);
}

// Create a deterministic input matrix using a simple pattern.
// Using fixed values ensures both PROTOCOL=1 and PROTOCOL=3 use the same input.
PlainMatrix<DataType> deterministicInput(size_t rows, size_t cols, DataType scale = 100) {
    cdough::Vector<DataType> vec(rows * cols);
    for (size_t i = 0; i < rows * cols; i++) {
        // Simple deterministic pattern: small values in fixed-point range
        vec[i] = ((DataType)(i % 97) - 48) * scale;
    }
    return PlainMatrix<DataType>(vec, rows, cols);
}

// Save results to file for cross-protocol comparison
void saveResults(EngineRef& engine, const std::string& filename, const std::string& label,
                 const PlainMatrix<DataType>& result) {
    // Only party 0 writes to avoid file corruption under MPI
    if (engine.getPartyID() != 0) return;

    std::ofstream file(filename, std::ios::app);
    auto data = result.data();
    file << label << " " << result.rows() << " " << result.cols() << " " << data.size() << "\n";
    for (size_t i = 0; i < data.size(); i++) {
        file << data[i];
        if (i < data.size() - 1) file << " ";
    }
    file << "\n";
    file.close();
}

// Load results from file
bool loadResults(const std::string& filename, const std::string& label,
                 std::vector<DataType>& values) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    while (std::getline(file, line)) {
        if (line.find(label) == 0) {
            // Found the label line, next line has the values
            if (std::getline(file, line)) {
                std::istringstream iss(line);
                DataType val;
                values.clear();
                while (iss >> val) {
                    values.push_back(val);
                }
                return true;
            }
        }
    }
    return false;
}

// Compare results from two protocol runs
void compareProtocolResults(EngineRef& engine, const std::string& file1, const std::string& file2,
                            const std::string& label, DataType tolerance) {
    std::vector<DataType> vals1, vals2;
    if (!loadResults(file1, label, vals1) || !loadResults(file2, label, vals2)) {
        single_cout("  " << label << ": Cannot compare (missing file)");
        return;
    }

    if (vals1.size() != vals2.size()) {
        single_cout("  " << label << ": SIZE MISMATCH " << vals1.size() << " vs " << vals2.size());
        return;
    }

    size_t mismatches = 0;
    DataType maxDiff = 0;

    for (size_t i = 0; i < vals1.size(); i++) {
        DataType diff = std::abs(vals1[i] - vals2[i]);
        if (diff > maxDiff) maxDiff = diff;
        if (diff > tolerance) {
            mismatches++;
            if (mismatches <= 3) {
                single_cout("    MISMATCH [" << i << "]: protocol1=" << vals1[i]
                            << " protocol3=" << vals2[i] << " diff=" << diff);
            }
        }
    }

    single_cout("  " << label << ": " << mismatches << "/" << vals1.size()
                << " mismatches (tolerance=" << tolerance << ", maxDiff=" << maxDiff << ")");
    if (mismatches == 0) {
        single_cout("  PASS: " << label);
    }
}

// Test: FullyConnected layer
template <typename Engine>
void testFC(Engine& engine, const std::string& weightsDir, size_t precision,
            const std::string& outputFile) {
    single_cout("\n  --- FullyConnected layer ---");

    const size_t inputDim = 512;
    const size_t outputDim = 10;

    // Load weights
    auto plainWeights = loadPlainMatrix(engine,
        weightsDir + "/classifier_6_weight.bin", inputDim, outputDim, true);
    auto plainBias = loadPlainMatrix(engine,
        weightsDir + "/classifier_6_bias.bin", 1, outputDim);

    // Deterministic input
    auto plainInput = deterministicInput(1, inputDim);

    // Secret share everything
    auto secInput = engine.secret_share_matrix(plainInput, 0);
    secInput.setPrecision(precision);
    auto secWeights = engine.secret_share_matrix(plainWeights, 0);
    secWeights.setPrecision(precision);
    auto secBias = engine.secret_share_matrix(plainBias, 0);
    secBias.setPrecision(precision);

    // Build model and run forward
    cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine> model(engine, precision);
    model.fullyConnectedLayerWithWeights(secWeights, secBias, inputDim, outputDim);
    auto output = model.forward(secInput);
    auto opened = output.open();

    single_cout("  Output: " << opened.rows() << "x" << opened.cols());
    auto data = opened.data();
    for (size_t i = 0; i < std::min((size_t)5, data.size()); i++) {
        single_cout("    [" << i << "] = " << data[i]);
    }

    saveResults(engine, outputFile, "FC", opened);
    single_cout("  Saved FC results to " << outputFile);
}

// Test: Conv2D layer
template <typename Engine>
void testConv2D(Engine& engine, const std::string& weightsDir, size_t precision,
                const std::string& outputFile) {
    single_cout("\n  --- Conv2D layer ---");

    const size_t inChannels = 3;
    const size_t outChannels = 64;
    const HW filterSize(3, 3);
    const HW inputSize(32, 32);
    const HW stride(1, 1);
    const HW padding(1, 1);

    const size_t filterRows = outChannels * filterSize.first;
    const size_t filterCols = filterSize.second * inChannels;

    auto plainFilter = loadPlainMatrix(engine,
        weightsDir + "/features_0_weight.bin", filterRows, filterCols);

    // Deterministic input: 32 rows, 96 cols (32 width * 3 channels)
    auto plainInput = deterministicInput(inputSize.first, inChannels * inputSize.second);

    auto secInput = engine.secret_share_matrix(plainInput, 0);
    secInput.setPrecision(precision);
    auto secFilter = engine.secret_share_matrix(plainFilter, 0);
    secFilter.setPrecision(precision);

    cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine> model(engine, precision);
    model.conv2DLayerWithWeights(secFilter, inputSize, inChannels, outChannels,
                                  filterSize, stride, padding);
    auto output = model.forward(secInput);
    auto opened = output.open();

    single_cout("  Output: " << opened.rows() << "x" << opened.cols());
    auto data = opened.data();
    for (size_t i = 0; i < std::min((size_t)5, data.size()); i++) {
        single_cout("    [" << i << "] = " << data[i]);
    }

    saveResults(engine, outputFile, "Conv2D", opened);
    single_cout("  Saved Conv2D results to " << outputFile);
}

// Test: ReLU layer
template <typename Engine>
void testReLU(Engine& engine, size_t precision, const std::string& outputFile) {
    single_cout("\n  --- ReLU layer ---");

    const size_t rows = 4;
    const size_t cols = 8;

    // Deterministic input with negative values (scale 4096 = 2^12, so values represent ~[-1.2, 1.2])
    auto plainInput = deterministicInput(rows, cols, 4096);

    auto secInput = engine.secret_share_matrix(plainInput, 0);
    secInput.setPrecision(precision);

    cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine> model(engine, precision);
    model.reLULayer(rows * cols);
    auto output = model.forward(secInput);
    auto opened = output.open();

    single_cout("  Output: " << opened.rows() << "x" << opened.cols());
    auto data = opened.data();
    for (size_t i = 0; i < std::min((size_t)5, data.size()); i++) {
        single_cout("    [" << i << "] = " << data[i]);
    }

    saveResults(engine, outputFile, "ReLU", opened);
    single_cout("  Saved ReLU results to " << outputFile);
}

// Test: AvgPooling layer
template <typename Engine>
void testAvgPool(Engine& engine, size_t precision, const std::string& outputFile) {
    single_cout("\n  --- AvgPooling layer ---");

    const HW inputSize(8, 8);
    const size_t inChannels = 4;
    const HW poolSize(2, 2);
    const HW stride(2, 2);
    const HW padding(0, 0);

    auto plainInput = deterministicInput(inputSize.first, inChannels * inputSize.second, 4096);

    auto secInput = engine.secret_share_matrix(plainInput, 0);
    secInput.setPrecision(precision);

    cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine> model(engine, precision);
    model.avgPoolingLayer(inputSize, inChannels, poolSize, stride, padding);
    auto output = model.forward(secInput);
    auto opened = output.open();

    single_cout("  Output: " << opened.rows() << "x" << opened.cols());
    auto data = opened.data();
    for (size_t i = 0; i < std::min((size_t)5, data.size()); i++) {
        single_cout("    [" << i << "] = " << data[i]);
    }

    saveResults(engine, outputFile, "AvgPool", opened);
    single_cout("  Saved AvgPool results to " << outputFile);
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);

    constexpr int PRECISION = 12;
    const std::string weightsDir = "weights/vgg16_cifar10";

    // Detect protocol from compiled macro
    int protocol = PROTOCOL_NUM;
    std::string outputFile = "layer_results_protocol" + std::to_string(protocol) + ".txt";

    // Clear output file (only party 0 writes files)
    if (engine.getPartyID() == 0) {
        std::ofstream(outputFile, std::ios::trunc).close();
    }

    single_cout("==============================================");
    single_cout("  Layer Test — Protocol " << protocol);
    single_cout("  Precision: " << PRECISION);
    single_cout("  Output: " << outputFile);
    single_cout("==============================================");

    // Run all layer tests and save results
    testFC(engine, weightsDir, PRECISION, outputFile);
    testConv2D(engine, weightsDir, PRECISION, outputFile);
    testReLU(engine, PRECISION, outputFile);
    testAvgPool(engine, PRECISION, outputFile);

    // If both protocol files exist, compare them
    std::string file1 = "layer_results_protocol1.txt";
    std::string file3 = "layer_results_protocol3.txt";

    std::ifstream check1(file1), check3(file3);
    if (check1.good() && check3.good()) {
        check1.close();
        check3.close();

        single_cout("\n==============================================");
        single_cout("  COMPARISON: Protocol 1 vs Protocol 3");
        single_cout("==============================================");

        // FC: 512 multiply-adds, tolerance for truncation rounding
        compareProtocolResults(engine, file1, file3, "FC", 512);
        // Conv2D: 27 multiply-adds per output
        compareProtocolResults(engine, file1, file3, "Conv2D", 27);
        // ReLU: no multiplication
        compareProtocolResults(engine, file1, file3, "ReLU", 0);
        // AvgPool: division rounding
        compareProtocolResults(engine, file1, file3, "AvgPool", 4);
    } else {
        single_cout("\n  To compare protocols, run under both PROTOCOL=1 and PROTOCOL=3.");
        single_cout("  Both result files will be compared automatically on the second run.");
    }

    single_cout("\n==============================================");
    single_cout("  Done.");
    single_cout("==============================================");

    return 0;
}
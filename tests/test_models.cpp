#include <numeric>

#include "cdough.h"

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;

using DataType = int64_t;
using HW = cdough::matrix::HeightWidth;

/**
 * @brief Test Model 1: Conv2D + ReLU
 * Simple model with a single convolution layer followed by ReLU activation.
 * Uses precomputed weights and ground truth values.
 */
template <typename Engine>
void test_model_conv2d_relu(Engine& engine) {
    auto pID = engine.getPartyID();
    const auto inputSize = HW(4, 4);
    const size_t inChannels = 1;
    const size_t outChannels = 1;
    const size_t inputCount = 1;
    const size_t precision = 0;  // we are testing layers composition

    single_cout_nonl("Testing Conv2D + ReLU... ");

    // Precomputed input: 4x4 matrix with some negative values to test ReLU
    cdough::Vector<DataType> inputData{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -13, -14, -15, -16};
    auto inputPlain = PlainMatrix<DataType>(inputData, 4, 4);
    auto input = engine.secret_share_matrix(inputPlain, 0);
    input.setPrecision(precision);

    // Precomputed filter: 2x2 filter
    cdough::Vector<DataType> filterData{1, 2, 3, 4};
    auto filterPlain = PlainMatrix<DataType>(filterData, 2, 2);
    auto filter = engine.secret_share_matrix(filterPlain, 0);
    filter.setPrecision(precision);

    // Ground truth: Conv2D with 2x2 filter, stride 1, no padding
    // Output size: (4-2)/1 + 1 = 3x3
    // Expected output (before ReLU):
    // 1*1+2*2+5*3+6*4 = 1+4+15+24 = 44
    // 2*1+3*2+6*3+7*4 = 2+6+18+28 = 54
    // 3*1+4*2+7*3+8*4 = 3+8+21+32 = 64
    // ... etc
    // After ReLU, negative values become 0,
    // so the last row becomes 0
    cdough::Vector<DataType> groundTruthData{44, 54, 64, 84, 94, 104, 0, 0, 0};
    auto groundTruth = PlainMatrix<DataType>(groundTruthData, 3, 3);

    // Build model with custom weights
    cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine> model(engine, precision);
    model.conv2DLayerWithWeights(filter, inputSize, inChannels, outChannels, HW(2, 2), HW(1, 1),
                                 HW(0, 0));
    model.reLULayer(inputCount * 3 * 3 * outChannels);

    // Run secure inference
    auto output_secure = model.forward(input);
    auto output_opened = output_secure.open();

    // if (pID == 0) output_opened.print();

    // Compare with ground truth
    if (pID == 0) {
        assert(output_opened.same_as(groundTruth));
        single_cout("OK");
    }
}

/**
 * @brief Test Model 2: Conv2D + AveragePool + ReLU
 * Model with convolution, average pooling, and ReLU activation.
 * Uses precomputed weights and ground truth values.
 */
template <typename Engine>
void test_model_conv2d_avgpool_relu(Engine& engine) {
    auto pID = engine.getPartyID();
    const auto inputSize = HW(4, 4);
    const size_t inChannels = 1;
    const size_t outChannels = 1;
    const size_t batchSize = 1;
    const size_t precision = 0;

    single_cout_nonl("Testing Conv2D + AveragePool + ReLU... ");

    // Precomputed input: 4x4 matrix
    cdough::Vector<DataType> inputData{1, 2, 3, 4,
                                    5, 6, 7, 8,
                                    9, 10, 11, 12,
                                    -13, -14, -30, -150};
    auto inputPlain = PlainMatrix<DataType>(inputData, 4, 4);
    auto input = engine.secret_share_matrix(inputPlain, 0);
    input.setPrecision(precision);

    // Precomputed filter: 2x2 filter
    cdough::Vector<DataType> filterData{1, 1, 1, 1, 1, 1, 1, 1, 1};
    auto filterPlain = PlainMatrix<DataType>(filterData, 3, 3);
    auto filter = engine.secret_share_matrix(filterPlain, 0);
    filter.setPrecision(precision);

    // Ground truth after Conv2D: 3x3
    // Conv output:
    // [
    //  [33      , 54      , 63      , 45      ],
    //  [3       , -9      , -140    , -142    ]
    // ]
    // After AvgePool:
    // [
    //  [20      , -44     ]
    // ]
    // After ReLU:
    // [
    //  [20      , 0       ]
    // ]
    cdough::Vector<DataType> groundTruthData{20, 0};
    auto groundTruth = PlainMatrix<DataType>(groundTruthData, 1, 2);

    // Build model with custom weights
    cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine> model(engine, precision);
    model.conv2DLayerWithWeights(filter, inputSize, inChannels, outChannels, HW(3, 3), HW(1, 1),
                                 HW(0, 1));
    model.avgPoolingLayer(HW(2, 4), outChannels, HW(2, 2), HW(2, 2), HW(0, 0));
    model.reLULayer(2);

    // Run secure inference
    auto output_secure = model.forward(input);
    auto output_opened = output_secure.open();

    // if (pID == 0) output_opened.print();

    // Compare with ground truth
    if (pID == 0) {
        assert(output_opened.same_as(groundTruth));
        single_cout("OK");
    }
}

/**
 * @brief Test Model 3: ReLU + Conv2D
 * Model with ReLU activation followed by convolution.
 * Uses precomputed weights and ground truth values.
 */
template <typename Engine>
void test_model_relu_conv2d(Engine& engine) {
    auto pID = engine.getPartyID();
    const auto inputSize = HW(4, 4);
    const size_t inChannels = 1;
    const size_t outChannels = 1;
    const size_t batchSize = 1;
    const size_t precision = 0;

    single_cout_nonl("Testing ReLU + Conv2D... ");

    // Precomputed input with negative values: 4x4 matrix
    cdough::Vector<DataType> inputData{-1, 2, -3, 4, -5, 6, -7, 8, 9, -10, 11, -12, 13, -14, 15, -16};
    auto inputPlain = PlainMatrix<DataType>(inputData, 4, 4);
    auto input = engine.secret_share_matrix(inputPlain, 0);
    input.setPrecision(precision);

    // After ReLU: [0, 2, 0, 4, 0, 6, 0, 8, 9, 0, 11, 0, 13, 0, 15, 0]
    // Precomputed filter: 2x2 filter with all 1's
    cdough::Vector<DataType> filterData{1, 1, 1, 1};
    auto filterPlain = PlainMatrix<DataType>(filterData, 2, 2);
    auto filter = engine.secret_share_matrix(filterPlain, 0);
    filter.setPrecision(precision);

    // Ground truth after ReLU + Conv2D: 3x3
    // After ReLU:
    // [
    //   [0       , 2       , 0       , 4       ],
    //   [0       , 6       , 0       , 8       ],
    //   [9       , 0       , 11      , 0       ],
    //   [13      , 0       , 15      , 0       ]
    // ]
    // Conv output:
    // [
    //   [8       , 8       , 12      ],
    //   [15      , 17      , 19      ],
    //   [22      , 26      , 26      ]
    // ]
    cdough::Vector<DataType> groundTruthData{8, 8, 12, 15, 17, 19, 22, 26, 26};
    auto groundTruth = PlainMatrix<DataType>(groundTruthData, 3, 3);

    // Build model with custom weights
    cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine> model(engine, precision);
    model.reLULayer(batchSize * inputSize.first * inputSize.second * inChannels);
    model.conv2DLayerWithWeights(filter, inputSize, inChannels, outChannels, HW(2, 2), HW(1, 1),
                                 HW(0, 0));

    // Run secure inference
    auto output_secure = model.forward(input);
    auto output_opened = output_secure.open();

    // if (pID == 0) output_opened.print();

    // Compare with ground truth
    if (pID == 0) {
        assert(output_opened.same_as(groundTruth));
        single_cout("OK");
    }
}

/**
 * @brief Test Model 4: Fully Connected Layer + ReLU
 * Model with a fully connected layer followed by ReLU activation.
 * Uses precomputed weights and ground truth values.
 */
template <typename Engine>
void test_model_fc_relu(Engine& engine) {
    auto pID = engine.getPartyID();
    const size_t inputDim = 4;
    const size_t outputDim = 3;
    const size_t precision = 0;
    const size_t batchSize = 2;

    single_cout_nonl("Testing Fully Connected + ReLU... ");

    // Precomputed input: 2x4 matrix (batch of 2)
    cdough::Vector<DataType> inputData{1, 2, 3, 4, 5, 6, 7, 8};
    auto inputPlain = PlainMatrix<DataType>(inputData, batchSize, inputDim);
    auto input = engine.secret_share_matrix(inputPlain, 0);
    input.setPrecision(precision);

    // Precomputed weights: 4x3 matrix (column-major)
    cdough::Vector<DataType> weightsData{-1, -2, -3, -4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto weightsPlain = PlainMatrix<DataType>(weightsData, inputDim, outputDim, true);
    auto weights = engine.secret_share_matrix(weightsPlain, 0);
    weights.setPrecision(precision);

    // Precomputed bias: 1x3 matrix
    cdough::Vector<DataType> biasData{1, 2, 3};
    auto biasPlain = PlainMatrix<DataType>(biasData, 1, outputDim, false);
    auto bias = engine.secret_share_matrix(biasPlain, 0);
    bias.setPrecision(precision);

    // Ground truth: FC layer output + bias + ReLU
    // FC output (before ReLU):
    // [
    //   [     -29,       72,      113],
    //   [     -69,      176,      281]
    // ]
    // After ReLU:
    // [
    //   [       0,       72,      113],
    //   [       0,      176,      281]
    // ]

    cdough::Vector<DataType> groundTruthData{0, 72, 113, 0, 176, 281};
    auto groundTruth = PlainMatrix<DataType>(groundTruthData, batchSize, outputDim);

    // Build model with custom weights
    cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine> model(engine, precision);
    model.fullyConnectedLayerWithWeights(weights, bias, inputDim, outputDim);
    model.reLULayer(batchSize * outputDim);

    // Run secure inference
    auto output_secure = model.forward(input);
    auto output_opened = output_secure.open();

    // if (pID == 0) output_opened.print();

    // Compare with ground truth
    if (pID == 0) {
        assert(output_opened.same_as(groundTruth));
        single_cout("OK");
    }
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();

    single_cout("\n============ Running Model Tests ============\n");

    // Test Model 1: Conv2D + ReLU
    test_model_conv2d_relu(engine);

    // Test Model 2: Conv2D + AveragePool + ReLU
    test_model_conv2d_avgpool_relu(engine);

    // Test Model 3: ReLU + Conv2D
    test_model_relu_conv2d(engine);

    // Test Model 4: Fully Connected + ReLU
    test_model_fc_relu(engine);

    single_cout("\n============ All Model Tests Passed! ============\n");

    return 0;
}
#pragma once

#include <fstream>
#include <map>
#include <stack>

#include "../containers/matrix/matrix.h"
#include "../containers/vector.h"

namespace cdough::operators::ml {

using cdough::matrix::HeightWidth;

////////////////////////////////////////////////////////////
///////////////// Weight File Loading /////////////////////
////////////////////////////////////////////////////////////

namespace {

    /**
     * @brief Load a raw binary weight file into a vector.
     *
     * Reads a .bin file produced by extract_weights.py (fixed-point int32,
     * little-endian, no headers) and returns the values as a std::vector<T>.
     *
     * @tparam T Data type (e.g., int64_t — values are widened from int32)
     * @param filepath Path to the .bin file
     * @return std::vector<T> The loaded weight values
     */
    template <typename T>
    std::vector<T> load_bin_file(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open weight file: " + filepath);
        }

        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (file_size % sizeof(int32_t) != 0) {
            throw std::runtime_error("Bad file size (not a multiple of 4 bytes): " + filepath);
        }

        size_t n = file_size / sizeof(int32_t);
        std::vector<int32_t> buf(n);
        file.read(reinterpret_cast<char*>(buf.data()), file_size);

        // Widen from int32 to T (e.g., int64_t) to match cdough's data type
        std::vector<T> result(n);
        for (size_t i = 0; i < n; i++) {
            result[i] = static_cast<T>(buf[i]);
        }
        return result;
    }

}  // anonymous namespace

////////////////////////////////////////////////////////////
/////////////////////// Layer Base /////////////////////////
////////////////////////////////////////////////////////////

/**
 * @brief Base class for Machine Learning layers
 *
 * @tparam T Data type
 * @tparam M Matrix type
 */
template <typename T, template <typename> class M>
class LayerML {
   public:
    LayerML() = default;
    virtual ~LayerML() = default;

    /**
     * @brief Forward pass of the layer
     * @param input Input matrix
     * @return M<T> Output matrix
     */
    virtual M<T> forward(const M<T>& input) = 0;
    // virtual M<T> backward(const M<T>& input) = 0;

    /**
     * @brief Get the name of the layer
     * @return std::string Name of the layer
     */
    virtual std::string name() const = 0;

    /**
     * @brief Load pretrained weights into this layer.
     *
     * Layers with learnable parameters (Conv2D, FullyConnected) override
     * this to accept weight matrices keyed by name ("filter", "weights",
     * "bias"). Layers without parameters (ReLU, AvgPool) use the default
     * no-op.
     *
     * @param weightMap Map of parameter name to matrix
     */
    virtual void loadWeights(const std::map<std::string, M<T>>& weightMap) {
        // Default no-op for layers without learnable weights
    }
};

////////////////////////////////////////////////////////////
////////////////////// Conv Layers /////////////////////////
////////////////////////////////////////////////////////////
/**
 * @brief Convolutional Layer
 *
 * @tparam T Data type
 * @tparam M Matrix type
 */
template <typename T, template <typename> class M>
class Conv2DLayer : public LayerML<T, M> {
   private:
    HeightWidth inputSize_;
    size_t inChannels_;
    size_t outChannels_;

    HeightWidth filterSize_;
    HeightWidth stride_;
    HeightWidth padding_;

    M<T> filter_;

   public:
    /**
     * @brief Construct a new Conv2D Layer object
     * @param filter Convolution filter matrix
     * @param inputSize Size of the input (height, width)
     * @param inChannels Number of input channels
     * @param outChannels Number of output channels
     * @param filterSize Size of the filter (height, width)
     * @param stride Stride of the convolution (height, width)
     * @param padding Padding of the convolution (height, width)
     */
    Conv2DLayer(const M<T>& filter, const HeightWidth& inputSize, size_t inChannels,
                size_t outChannels, const HeightWidth& filterSize, const HeightWidth& stride,
                const HeightWidth& padding)
        : inputSize_(inputSize),
          inChannels_(inChannels),
          outChannels_(outChannels),
          filterSize_(filterSize),
          stride_(stride),
          padding_(padding),
          filter_(filter) {}

    /**
     * @brief Forward pass of the Conv2D layer
     * @param input Input matrix
     * @return M<T> Output matrix
     */
    M<T> forward(const M<T>& input) override {
        const auto filterSizeWithChannels =
            HeightWidth(filterSize_.first, filterSize_.second * inChannels_);
        const auto strideWithChannels = HeightWidth(stride_.first, stride_.second * inChannels_);
        const auto paddingWithChannels = HeightWidth(padding_.first, padding_.second * inChannels_);
        const auto batchSize = input.rows() / inputSize_.first;

#ifdef PRINT_ML_LAYERS_DIMENSIONS
        std::cout << "Conv2DLayer: inputSize = " << input.rows() << "x" << input.cols()
                  << ", inChannels = " << inChannels_ << ", outChannels = " << outChannels_
                  << ", filterSize = " << filterSize_.first << "x" << filterSize_.second
                  << ", stride = " << stride_.first << "x" << stride_.second
                  << ", padding = " << padding_.first << "x" << padding_.second << std::endl;
#endif  // PRINT_ML_LAYERS_DIMENSIONS

        return input.conv2DVectorized(filter_, batchSize, filterSizeWithChannels,
                                      strideWithChannels, paddingWithChannels);
    }

    /**
     * @brief Get the name of the layer
     * @return std::string Name of the layer "Conv2D"
     */
    std::string name() const override { return "Conv2D"; }

    /**
     * @brief Load pretrained filter weights.
     * @param weightMap Must contain "filter" key
     */
    void loadWeights(const std::map<std::string, M<T>>& weightMap) override {
        if (weightMap.count("filter")) {
            filter_ = weightMap.at("filter");
        }
    }
};

/**
 * @brief Fully Connected Layer
 *
 * @tparam T Data type
 * @tparam M Matrix type
 */
template <typename T, template <typename> class M>
class FullyConnectedLayer : public LayerML<T, M> {
   private:
    size_t inputDim_;
    size_t outputDim_;

    M<T> weights_;
    M<T> bias_;

   public:
    /**
     * @brief Construct a new Fully Connected Layer object
     *
     * @param weights Weights matrix
     * @param bias Bias matrix
     * @param inputDim Input dimension
     * @param outputDim Output dimension
     */
    FullyConnectedLayer(const M<T>& weights, const M<T>& bias, size_t inputDim, size_t outputDim)
        : inputDim_(inputDim), outputDim_(outputDim), weights_(weights), bias_(bias) {}

    /**
     * @brief Forward pass of the Fully Connected layer
     *
     * @param input Input matrix
     * @return M<T> Output matrix
     */
    M<T> forward(const M<T>& input) override {
#ifdef PRINT_ML_LAYERS_DIMENSIONS
        std::cout << "FullyConnectedLayer: inputSize = " << input.rows() << "x" << input.cols()
                  << ", inputDim = " << inputDim_ << ", outputDim = " << outputDim_ << std::endl;
#endif  // PRINT_ML_LAYERS_DIMENSIONS
        auto instancesCount = input.data().size() / inputDim_;
        auto input_ = input.reshapeRef(instancesCount, inputDim_);
        return input_.fullyConnectedVectorized(weights_, bias_);
    }

    /**
     * @brief Get the name of the layer
     * @return std::string Name of the layer "Fully"
     */
    std::string name() const override { return "Fully "; }

    /**
     * @brief Load pretrained weights and bias.
     * @param weightMap Must contain "weights" and "bias" keys
     */
    void loadWeights(const std::map<std::string, M<T>>& weightMap) override {
        if (weightMap.count("weights")) {
            weights_ = weightMap.at("weights");
        }
        if (weightMap.count("bias")) {
            bias_ = weightMap.at("bias");
        }
    }
};

////////////////////////////////////////////////////////////
//////////////////// Pooling Layers ////////////////////////
////////////////////////////////////////////////////////////
/**
 * @brief Average Pooling Layer
 *
 * @tparam T Data type
 * @tparam M Matrix type
 */
template <typename T, template <typename> class M>
class AvgPoolingLayer : public LayerML<T, M> {
   private:
    HeightWidth inputSize_;
    size_t inChannels_;
    HeightWidth filterSize_;
    HeightWidth stride_;
    HeightWidth padding_;

   public:
    /**
     * @brief Construct a new Avg Pooling Layer object
     *
     * @param inputSize Size of the input (height, width)
     * @param inChannels Number of input channels
     * @param filterSize Size of the filter (height, width)
     * @param stride Stride of the pooling (height, width)
     * @param padding Padding of the pooling (height, width)
     */
    AvgPoolingLayer(const HeightWidth& inputSize, size_t inChannels, const HeightWidth& filterSize,
                    const HeightWidth& stride, const HeightWidth& padding)
        : inputSize_(inputSize),
          inChannels_(inChannels),
          filterSize_(filterSize),
          stride_(stride),
          padding_(padding) {}

    /**
     * @brief Forward pass of the Avg Pooling layer
     * @param input Input matrix
     * @return M<T> Output matrix
     */
    M<T> forward(const M<T>& input) override {
#ifdef PRINT_ML_LAYERS_DIMENSIONS
        std::cout << "AvgPoolingLayer: inputSize = " << input.rows() << "x" << input.cols()
                  << ", inChannels = " << inChannels_ << ", filterSize = " << filterSize_.first
                  << "x" << filterSize_.second << ", stride = " << stride_.first << "x"
                  << stride_.second << ", padding = " << padding_.first << "x" << padding_.second
                  << std::endl;
#endif  // PRINT_ML_LAYERS_DIMENSIONS

        // Apply average pooling
        const auto batchSize = input.rows() / inputSize_.first;
        M<T> result = input.avgPoolingVectorized(batchSize, inChannels_, inputSize_, filterSize_,
                                                 stride_, padding_);
        return result;
    }

    /**
     * @brief Get the name of the layer
     * @return std::string Name of the layer "Avg"
     */
    std::string name() const override { return "Avg   "; }
};

////////////////////////////////////////////////////////////
///////////////// Activation Functions /////////////////////
////////////////////////////////////////////////////////////
/**
 * @brief ReLU Activation Layer
 *
 * @tparam T Data type
 * @tparam M Matrix type
 */
template <typename T, template <typename> class M>
class ReLULayer : public LayerML<T, M> {
    size_t inputDim;

   public:
    /**
     * @brief Construct a new ReLU Layer object
     *
     * @param inputDim Input dimension
     */
    ReLULayer(size_t inputDim) : inputDim(inputDim) {}

    /**
     * @brief Forward pass of the ReLU layer
     *
     * @param input Input matrix
     * @return M<T> Output matrix
     */
    M<T> forward(const M<T>& input) override {
#ifdef PRINT_ML_LAYERS_DIMENSIONS
        std::cout << "ReLULayer: inputSize = " << input.rows() << "x" << input.cols()
                  << ", vectorDim: " << input.rows() * input.cols() << ", inputDim = " << inputDim
                  << std::endl;
#endif  // PRINT_ML_LAYERS_DIMENSIONS
        return input.reLUVectorized();
    }

    /**
     * @brief Get the name of the layer
     * @return std::string Name of the layer "ReLU"
     */
    std::string name() const override { return "ReLU  "; }
};

////////////////////////////////////////////////////////////
///////////////////////// Model ////////////////////////////
////////////////////////////////////////////////////////////
/**
 * @brief Forward pass through the model layers recursively
 *
 * @param input Input matrix
 * @param layers Vector of layer pointers
 * @param index Current layer index
 * @return M<T> Output matrix
 */
template <typename T, template <typename> class M>
M<T> forwardRecursive(const M<T>& input, const std::vector<std::shared_ptr<LayerML<T, M>>>& layers,
                      size_t index = 0) {
    if (index == layers.size()) {
        return input;
    }
    return forwardRecursive<T, M>(layers[index]->forward(input), layers, index + 1);
}

/**
 * @brief Forward pass through the model layers iteratively
 *
 * @param input Input matrix
 * @param layers Vector of layer pointers
 * @param index Starting layer index
 * @return M<T> Output matrix
 */
template <typename T, template <typename> class M, typename Engine>
M<T> forwardIterative(const M<T>& input, const std::vector<std::shared_ptr<LayerML<T, M>>>& layers,
                    Engine& engine, size_t index = 0) {
    std::stack<M<T>> inputStack;
    inputStack.push(input);

#ifdef PROFILE_ML_LAYERS
    stopwatch::timepoint("Forward Start");
#endif  // PROFILE_ML_LAYERS

    while (index < layers.size()) {
        M<T> currentInput = inputStack.top();
        inputStack.pop();

        std::string reluLaerName = "ReLU  ";
        if (false && layers[index]->name() == reluLaerName) {
            // we will change batch size to -1 if WAN
            auto setting = engine.getSetting();
            if (setting == service::Setting::WAN) {
                auto old_batch_size = engine.get_batch_size();
                engine.setBatchSize(-1);  // set batch size to -1 for ReLU
                M<T> output = layers[index]->forward(currentInput);
                inputStack.push(output);
                engine.setBatchSize(old_batch_size);  // restore original batch size
            }else{
                M<T> output = layers[index]->forward(currentInput);
                inputStack.push(output);
            }
        }else{
            M<T> output = layers[index]->forward(currentInput);
            inputStack.push(output);
        }

#ifdef PROFILE_ML_LAYERS
        stopwatch::timepoint("L" + std::to_string(index) + " " + layers[index]->name());
#endif  // PROFILE_ML_LAYERS

        index++;
    }

    return inputStack.top();
}

/**
 * @brief Machine Learning Model
 *
 * The model supports adding layers and performing
 * a forward pass through the layers.
 *
 * Layer weights have channels in interleaved format.
 * Inputs have the same interleaved channel format.
 * Inputs can be concatenated for multiple instances.
 *
 * For example, to do inference over a batch of 3 instances,:
 * [
 *  Instance1_ROWS
 *  Instance2_ROWS
 *  Instance3_ROWS
 * ]
 *
 * Note for each instance:
 * [
 * ROW1: ch1_val, ch2_val, ch3_val, ... ch1_val, ch2_val, ch3_val
 * ROW2: ch1_val, ch2_val, ch3_val, ... ch1_val, ch2_val, ch3_val
 * ...
 * ]
 *
 *
 * @tparam T Data type
 * @tparam M Matrix type
 * @tparam Engine Engine type
 */
template <typename T, template <typename> class M, typename Engine>
class ModelML {
   private:
    std::vector<std::shared_ptr<LayerML<T, M>>> layers;
    const size_t precision;
    Engine& engine;

   public:
    /**
     * @brief Construct a new Model ML object
     *
     * @param engine Engine reference
     * @param precision Precision for fixed-point representation
     */
    ModelML(Engine& engine, size_t precision) : engine(engine), precision(precision) {}
    ~ModelML() = default;

    /**
     * @brief Adds an already constructed layer to the model.
     *
     * @param layer Layer pointer
     */
    void addLayer(std::shared_ptr<LayerML<T, M>> layer) { layers.push_back(layer); }

    /**
     * @brief Forward pass through the model
     * to run inference on multiple inputs.
     *
     * @param input Input matrix
     * @return M<T> Output matrix
     */
    M<T> forward(const M<T>& input) { return forwardIterative<T, M>(input, layers, engine, 0); }

    /**
     * @brief Adds a Conv2D layer to the model with uninitialized weights.
     *
     * @param inputSize Size of the input (height, width) for each channel
     * @param inChannels Number of input channels
     * @param outChannels Number of output channels
     * @param filterSize Size of the filter (height, width)
     * @param stride Stride of the convolution (height, width)
     * @param padding Padding of the convolution (height, width)
     * @return ModelML& Reference to the model
     */
    ModelML& conv2DLayer(const HeightWidth& inputSize, size_t inChannels, size_t outChannels,
                         const HeightWidth& filterSize, const HeightWidth& stride,
                         const HeightWidth& padding) {
        auto filter = engine.template public_share_zero_matrix<T>(
            outChannels * filterSize.first, filterSize.second * inChannels, false);
        filter.setPrecision(precision);
        layers.push_back(std::make_shared<Conv2DLayer<T, M>>(
            filter, inputSize, inChannels, outChannels, filterSize, stride, padding));
        return *this;
    }

    /**
     * @brief Adds a Fully Connected layer to the model with random weights.
     *
     * @param inputDim Input dimension
     * @param outputDim Output dimension
     * @return ModelML& Reference to the model
     */
    ModelML& fullyConnectedLayer(size_t inputDim, size_t outputDim) {
        M<T> weights = engine.template public_share_zero_matrix<T>(inputDim, outputDim, true);
        weights.setPrecision(precision);
        M<T> bias = engine.template public_share_zero_matrix<T>(1, outputDim, false);
        bias.setPrecision(precision);
        layers.push_back(
            std::make_shared<FullyConnectedLayer<T, M>>(weights, bias, inputDim, outputDim));
        return *this;
    }

    /**
     * @brief Adds an Average Pooling layer to the model.
     *
     * @param inputSize Size of the input (height, width)
     * @param inChannels Number of input channels
     * @param filterSize Size of the filter (height, width)
     * @param stride Stride of the pooling (height, width)
     * @param padding Padding of the pooling (height, width)
     * @return ModelML& Reference to the model
     */
    ModelML& avgPoolingLayer(const HeightWidth& inputSize, size_t inChannels,
                             const HeightWidth& filterSize, const HeightWidth& stride,
                             const HeightWidth& padding) {
        layers.push_back(std::make_shared<AvgPoolingLayer<T, M>>(inputSize, inChannels, filterSize,
                                                                 stride, padding));
        return *this;
    }

    /**
     * @brief Adds a ReLU activation layer to the model.
     *
     * @param inputDim Input dimension
     * @return ModelML& Reference to the model
     */
    ModelML& reLULayer(size_t inputDim) {
        layers.push_back(std::make_shared<ReLULayer<T, M>>(inputDim));
        return *this;
    }

    ////////////////////////////////////////////////////////////
    //////////// Layer Loading from Pretrained Weights //////////
    ////////////////////////////////////////////////////////////

    /**
     * @brief Adds a Conv2D layer with a pretrained filter matrix.
     *
     * Use load_bin_file() to load the raw data, construct a PlainMatrix,
     * and secret share it in the calling code (bench file), then pass
     * the resulting SecureMatrix here.
     *
     * @param filter Pretrained filter matrix (already secret shared)
     * @param inputSize Size of the input (height, width) for each channel
     * @param inChannels Number of input channels
     * @param outChannels Number of output channels
     * @param filterSize Size of the filter (height, width)
     * @param stride Stride of the convolution (height, width)
     * @param padding Padding of the convolution (height, width)
     * @return ModelML& Reference to the model
     */
    ModelML& conv2DLayerWithWeights(const M<T>& filter, const HeightWidth& inputSize,
                                    const size_t& inChannels, const size_t& outChannels,
                                    const HeightWidth& filterSize, const HeightWidth& stride,
                                    const HeightWidth& padding) {
        layers.push_back(std::make_shared<Conv2DLayer<T, M>>(
            filter, inputSize, inChannels, outChannels, filterSize, stride, padding));
        return *this;
    }

    /**
     * @brief Adds a Fully Connected layer with pretrained weight and bias matrices.
     *
     * Use load_bin_file() to load the raw data, construct PlainMatrix objects,
     * and secret share them in the calling code (bench file), then pass
     * the resulting SecureMatrices here.
     *
     * @param weights Pretrained weight matrix (already secret shared)
     * @param bias Pretrained bias matrix (already secret shared)
     * @param inputDim Input dimension
     * @param outputDim Output dimension
     * @return ModelML& Reference to the model
     */
    ModelML& fullyConnectedLayerWithWeights(const M<T>& weights, const M<T>& bias,
                                            const size_t& inputDim, const size_t& outputDim) {
        layers.push_back(
            std::make_shared<FullyConnectedLayer<T, M>>(weights, bias, inputDim, outputDim));
        return *this;
    }
};
}  // namespace cdough::operators::ml
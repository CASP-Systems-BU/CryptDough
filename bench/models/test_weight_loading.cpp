/**
 * Models tested:
 * AlexNet (ImageNet): torchvision pretrained
 * VGG16-CIFAR10 (CIFAR-10): chenyaofo/pytorch-cifar-models, 94.16% acc
 * VGG16-ImageNet (ImageNet): torchvision pretrained
 *
 * The test: load .bin -> cdough Vector -> secret_share_a -> open -> compare
 *
 * Plaintext: cmake .. -DPROTOCOL=1 && make test_weight_loading && ./test_weight_loading
 * 3PC MPC: cmake .. -DPROTOCOL=3 && make test_weight_loading && mpirun -np 3 ./test_weight_loading
 */

#include "cdough.h"

using namespace cdough::debug;
using namespace cdough::service;
using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

// Load a raw .bin file into an cdough Vector<int>
cdough::Vector<int> load_bin(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return cdough::Vector<int>(0);
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size % sizeof(int32_t) != 0) {
        std::cerr << "  ERROR: Bad file size: " << filepath << std::endl;
        return cdough::Vector<int>(0);
    }

    size_t n = file_size / sizeof(int32_t);
    std::vector<int32_t> buf(n);
    file.read(reinterpret_cast<char*>(buf.data()), file_size);

    cdough::Vector<int> vec(n);
    for (size_t i = 0; i < n; i++) {
        vec[i] = buf[i];
    }
    return vec;
}

// Test one weight file: load -> secret share -> open -> compare
// Returns: 0 = pass, 1 = fail, -1 = skipped
template <typename Engine>
int test_one_weight(Engine& engine, const std::string& filepath,
                    const std::string& label, int precision) {
    cdough::Vector<int> original = load_bin(filepath);
    if (original.size() == 0) {
        return -1;
    }

    ASharedVector<int> shared = engine.secret_share_a(original, 0);
    shared.setPrecision(precision);

    auto reconstructed = shared.open();

    int mismatches = 0;
    for (size_t i = 0; i < original.size(); i++) {
        if (original[i] != reconstructed[i]) {
            mismatches++;
            if (mismatches <= 3) {
                std::cerr << "    MISMATCH at [" << i << "]: original="
                          << original[i] << " reconstructed=" << reconstructed[i] << std::endl;
            }
        }
    }

    if (mismatches == 0) {
        single_cout("  PASS  " << label << "  (" << original.size() << " values)");
        return 0;
    } else {
        single_cout("  FAIL  " << label << "  (" << mismatches << "/"
                    << original.size() << " mismatches)");
        return 1;
    }
}

// Test all weight files for one model
template <typename Engine>
bool test_model(Engine& engine, const std::string& model_dir,
                const std::string& model_name, int precision) {

    single_cout("\n----------------------------------------------");
    single_cout("  Model: " << model_name);
    single_cout("  Dir:   " << model_dir);
    single_cout("----------------------------------------------");

    // Try all plausible layer indices for features (0-43) and classifier (0-6).
    // Both weight, bias, running_mean, running_var.
    // Missing files are silently skipped — this handles both architectures:
    //   ImageNet VGG16:  features.0,2,5,7,10,12,14,17,19,21,24,26,28 (no BN)
    //   CIFAR-10 VGG16:  features.0,1,3,4,7,8,10,11,14,15,17,18,20,21,24,25,27,28,30,31,34,35,37,38,40,41 (with BN)
    //   AlexNet:         features.0,3,6,8,10; classifier.1,4,6

    std::vector<std::pair<std::string, std::string>> files;

    // Generate all plausible feature layer files
    for (int i = 0; i <= 43; i++) {
        std::string idx = std::to_string(i);
        files.push_back({"features_" + idx + "_weight.bin",       "features." + idx + " weight"});
        files.push_back({"features_" + idx + "_bias.bin",         "features." + idx + " bias"});
        files.push_back({"features_" + idx + "_running_mean.bin", "features." + idx + " running_mean"});
        files.push_back({"features_" + idx + "_running_var.bin",  "features." + idx + " running_var"});
    }

    // Generate all plausible classifier layer files
    for (int i = 0; i <= 6; i++) {
        std::string idx = std::to_string(i);
        files.push_back({"classifier_" + idx + "_weight.bin", "classifier." + idx + " weight"});
        files.push_back({"classifier_" + idx + "_bias.bin",   "classifier." + idx + " bias"});
    }

    int passed = 0, failed = 0, skipped = 0;

    for (auto& [filename, label] : files) {
        std::string path = model_dir + "/" + filename;
        int result = test_one_weight(engine, path, label, precision);
        if (result == 0) passed++;
        else if (result == 1) failed++;
        else skipped++;
    }

    single_cout("  Summary: " << passed << " passed, "
                << failed << " failed, " << skipped << " skipped");

    return (failed == 0 && passed > 0);
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();

    constexpr int PRECISION = 12;

    single_cout("==============================================");
    single_cout("  PyTorch Weight Import Test");
    single_cout("  Testing: load -> secret share -> open");
    single_cout("  Precision: " << PRECISION);
    single_cout("==============================================");

    bool p1 = test_model(engine, "weights/alexnet_imagenet", "AlexNet (ImageNet)",   PRECISION);
    bool p2 = test_model(engine, "weights/vgg16_cifar10", "VGG16-BN (CIFAR-10)",  PRECISION);
    bool p3 = test_model(engine, "weights/vgg16_imagenet", "VGG16 (ImageNet)",     PRECISION);

    single_cout("\n==============================================");
    single_cout("  RESULTS:");
    single_cout("    AlexNet (ImageNet): " << (p1 ? "PASS" : "FAIL"));
    single_cout("    VGG16-BN (CIFAR-10): " << (p2 ? "PASS" : "FAIL"));
    single_cout("    VGG16 (ImageNet): " << (p3 ? "PASS" : "FAIL"));
    single_cout("==============================================");

    return (p1 && p2 && p3) ? 0 : 1;
}
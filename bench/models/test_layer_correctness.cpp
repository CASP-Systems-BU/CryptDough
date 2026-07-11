/**
 * test_layer_correctness.cpp
 *
 * Compares cdough layer output against PyTorch ground truth at
 * increasing sizes and precisions. Uses isolation tests to
 * identify which conv2D parameters cause mismatches.
 *
 * Run generate_layer_test_data.py first to create test_data/.
 *
 * Build:
 *   cmake .. -DPROTOCOL=3 && make test_layer_correctness
 *   mpirun -np 3 ./test_layer_correctness
 */

#include "cdough.h"
#include <fstream>
#include <cmath>
#include <vector>
#include <string>

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;

using DataType = int64_t;
using HW = cdough::matrix::HeightWidth;

PlainMatrix<DataType> loadPlain(EngineRef& engine, const std::string& path,
                                size_t rows, size_t cols, bool colWise = false) {
    auto raw = cdough::operators::ml::load_bin_file<DataType>(path);
    if (raw.size() != rows * cols)
        throw std::runtime_error("size mismatch in " + path);
    cdough::Vector<DataType> v(raw.size());
    for (size_t i = 0; i < raw.size(); i++) v[i] = raw[i];
    return PlainMatrix<DataType>(v, rows, cols, colWise);
}

std::vector<DataType> loadExpected(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("cannot open " + path);
    size_t nbytes = f.tellg();
    f.seekg(0);
    size_t n = nbytes / sizeof(int64_t);
    std::vector<int64_t> buf(n);
    f.read(reinterpret_cast<char*>(buf.data()), nbytes);
    return std::vector<DataType>(buf.begin(), buf.end());
}

size_t readMeta(const std::string& path, const std::string& key) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("cannot open " + path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.find(key + "=") == 0)
            return std::stoull(line.substr(key.size() + 1));
    }
    throw std::runtime_error("key not found: " + key);
}

// comparison
struct Result {
    std::string label;
    size_t elements = 0;
    size_t mismatches = 0;
    DataType maxDiff = 0;
    double rmse = 0;
    bool passed = false;
};

Result compare(EngineRef& engine, const std::string& label,
               const std::vector<DataType>& actual,
               const std::vector<DataType>& expected) {
    Result r;
    r.label = label;
    r.elements = actual.size();

    if (actual.size() != expected.size()) {
        single_cout("  SIZE MISMATCH: got " << actual.size()
                    << ", expected " << expected.size());
        return r;
    }

    double sse = 0.0;
    for (size_t i = 0; i < actual.size(); i++) {
        DataType d = actual[i] - expected[i];
        DataType ad = std::abs(d);
        if (ad > r.maxDiff) r.maxDiff = ad;
        sse += (double)d * (double)d;
        if (ad > 0) {
            r.mismatches++;
            if (r.mismatches <= 5)
                single_cout("    [" << i << "] cdough=" << actual[i]
                            << " expected=" << expected[i] << " diff=" << d);
        }
    }

    r.rmse = std::sqrt(sse / actual.size());
    r.passed = (r.mismatches == 0);

    single_cout("    " << r.mismatches << "/" << r.elements
                << " mismatches, maxDiff=" << r.maxDiff
                << ", RMSE=" << r.rmse
                << " -> " << (r.passed ? "PASS" : "FAIL"));
    return r;
}

// layer tests
template <typename Engine>
Result runFC(Engine& engine, const std::string& dir, bool verbose) {
    std::string meta = dir + "/fc_meta.txt";
    size_t inDim  = readMeta(meta, "inputDim");
    size_t outDim = readMeta(meta, "outputDim");
    size_t prec   = readMeta(meta, "precision");

    single_cout("  FC " << inDim << "x" << outDim << " prec=" << prec);

    auto inp = loadPlain(engine, dir + "/fc_input.bin", 1, inDim);
    auto w   = loadPlain(engine, dir + "/fc_weight.bin", inDim, outDim, true);
    auto b   = loadPlain(engine, dir + "/fc_bias.bin", 1, outDim);
    auto exp = loadExpected(dir + "/fc_expected_output.bin");

    if (verbose) {
        single_cout("    input: " << inDim << " values, weight: "
                    << inDim << "x" << outDim << " (colWise)");
        for (size_t i = 0; i < exp.size(); i++)
            single_cout("    expected[" << i << "] = " << exp[i]);
    }

    auto sInp = engine.secret_share_matrix(inp, 0); sInp.setPrecision(prec);
    auto sW   = engine.secret_share_matrix(w, 0);   sW.setPrecision(prec);
    auto sB   = engine.secret_share_matrix(b, 0);   sB.setPrecision(prec);

    cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine> model(engine, prec);
    model.fullyConnectedLayerWithWeights(sW, sB, inDim, outDim);

    auto out = model.forward(sInp).open();
    auto od = out.data();
    std::vector<DataType> actual(od.begin(), od.end());

    if (verbose) {
        for (size_t i = 0; i < actual.size(); i++)
            single_cout("    cdough[" << i << "] = " << actual[i]);
    }

    return compare(engine, "FC", actual, exp);
}

template <typename Engine>
Result runConv(Engine& engine, const std::string& dir, bool verbose) {
    std::string meta = dir + "/conv_meta.txt";
    size_t inCh  = readMeta(meta, "inChannels");
    size_t outCh = readMeta(meta, "outChannels");
    size_t kH    = readMeta(meta, "kH");
    size_t kW    = readMeta(meta, "kW");
    size_t iH    = readMeta(meta, "inputH");
    size_t iW    = readMeta(meta, "inputW");
    size_t s     = readMeta(meta, "stride");
    size_t pad   = readMeta(meta, "padding");
    size_t prec  = readMeta(meta, "precision");

    single_cout("  Conv2D " << iH << "x" << iW << " " << inCh << "ch->" << outCh
                << "ch, " << kH << "x" << kW << " filt, pad=" << pad << " prec=" << prec);

    auto inp = loadPlain(engine, dir + "/conv_input.bin", iH, iW * inCh);
    auto flt = loadPlain(engine, dir + "/conv_filter.bin", outCh * kH, kW * inCh);
    auto exp = loadExpected(dir + "/conv_expected_output.bin");

    auto sInp = engine.secret_share_matrix(inp, 0);  sInp.setPrecision(prec);
    auto sFlt = engine.secret_share_matrix(flt, 0);   sFlt.setPrecision(prec);

    cdough::operators::ml::ModelML<DataType, SecureMatrix, Engine> model(engine, prec);
    model.conv2DLayerWithWeights(sFlt, HW(iH, iW), inCh, outCh,
                                  HW(kH, kW), HW(s, s), HW(pad, pad));

    auto out = model.forward(sInp).open();
    single_cout("    output: " << out.rows() << "x" << out.cols());
    auto od = out.data();
    std::vector<DataType> actual(od.begin(), od.end());

    return compare(engine, "Conv2D", actual, exp);
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    std::string base = "test_data";

    single_cout("=== Layer Correctness — Protocol " << PROTOCOL_NUM << " ===\n");

    std::vector<std::pair<std::string, Result>> results;

    // p=0 baseline (must be exact match)
    single_cout("-- p=0 baseline --");
    results.push_back({"P1 FC  (p=0 tiny)",  runFC(engine, base + "/fc_p0_tiny", true)});
    results.push_back({"P1 Conv baseline",    runConv(engine, base + "/conv_p0_tiny", true)});

    // Isolation: change one variable at a time from baseline
    // baseline = 3x3 in, 2ch->2ch, 2x2 filt, pad=0
    single_cout("\n-- Isolation (p=0, one change each) --");
    results.push_back({"ISO-A inCh=3",     runConv(engine, base + "/conv_p0_iso_3inch", false)});
    results.push_back({"ISO-B outCh=4",    runConv(engine, base + "/conv_p0_iso_4outch", false)});
    results.push_back({"ISO-C filt=3x3",   runConv(engine, base + "/conv_p0_iso_3x3filt", false)});
    results.push_back({"ISO-D pad=1",      runConv(engine, base + "/conv_p0_iso_pad1", false)});
    results.push_back({"ISO-E size=8x8",   runConv(engine, base + "/conv_p0_iso_8x8", false)});
    results.push_back({"ISO-F med nopad",  runConv(engine, base + "/conv_p0_medium_nopad", false)});

    // p=0 full medium
    single_cout("\n-- p=0 medium --");
    results.push_back({"P2 FC  (p=0 med)",  runFC(engine, base + "/fc_p0_medium", false)});
    results.push_back({"P2 Conv (p=0 med)", runConv(engine, base + "/conv_p0_medium", false)});

    // increasing precision
    single_cout("\n-- p=4 tiny --");
    results.push_back({"P3 FC  (p=4 tiny)",  runFC(engine, base + "/fc_p4_tiny", false)});
    results.push_back({"P3 Conv (p=4 tiny)", runConv(engine, base + "/conv_p4_tiny", false)});

    single_cout("\n-- p=8 medium --");
    results.push_back({"P4 FC  (p=8 med)",  runFC(engine, base + "/fc_p8_medium", false)});
    results.push_back({"P4 Conv (p=8 med)", runConv(engine, base + "/conv_p8_medium", false)});

    single_cout("\n-- p=12 medium --");
    results.push_back({"P5 FC  (p=12 med)",  runFC(engine, base + "/fc_p12_medium", false)});
    results.push_back({"P5 Conv (p=12 med)", runConv(engine, base + "/conv_p12_medium", false)});

    single_cout("\n=== SUMMARY ===\n");
    single_cout("  Test                   | Elems | Mismatch | MaxDiff |    RMSE | Result");
    single_cout("  -----------------------|-------|----------|---------|---------|-------");

    for (auto& [name, r] : results) {
        char line[200];
        snprintf(line, sizeof(line), "  %-23s| %5zu | %8zu | %7lld | %7.1f | %s",
            name.c_str(), r.elements, r.mismatches,
            (long long)r.maxDiff, r.rmse, r.passed ? "PASS" : "FAIL");
        single_cout(line);
    }

    single_cout("\n  Isolation breakdown:");
    for (auto& [name, r] : results) {
        if (name.find("ISO-") == std::string::npos) continue;
        single_cout("    " << name << ": " << (r.passed ? "OK" : "BROKE"));
    }

    single_cout("");
    return 0;
}
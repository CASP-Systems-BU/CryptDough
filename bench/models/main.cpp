
#include <numeric>

#include "cdough.h"

// python3 ../scripts/run_experiment.py -p 3 -r 16 alexnet

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;
using namespace cdough::debug;
using namespace cdough::service;

using DataType = int64_t;
using HW = cdough::matrix::HeightWidth;

#include <bit>
#include <bitset>
#include <cstdint>
#include <iostream>

static std::string float_to_ieee754(float x) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));  // C++17-safe bit reinterpretation

    std::bitset<32> b(bits);
    std::string s = b.to_string();

    // s eeeeeeee mmmmmmmmmmmmmmmmmmmmmmm
    return s.substr(0, 1) + " " + s.substr(1, 8) + " " + s.substr(9, 23);
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();

    // for(int x = 0; x < 2; x++) {
    //     for(int y = 0; y < 2; y++) {
    //         for(int c = 0; c < 2; c++) {
    //             auto res = x*y + x*c + y*c - 2*(x*y*c);
    //             auto res2 = x + y + c - 2*(x*y + x*c + y*c) + 4*x*y*c;

    //             // std::cout << "x: " << x << ", y: " << y << ", c: " << c << " => res: " << res << std::endl;
    //             std::cout << "x: " << x << ", y: " << y << ", c: " << c << " => res2: " << res2 << std::endl;
    //         }
    //     }
    // }

    float x = 1.0f / 5.0f;  // 0.2f
    float y = 1.0f;  // 1.0f
    std::cout << x << " -> " << float_to_ieee754(x) << '\n';
    std::cout << y << " -> " << float_to_ieee754(y) << '\n';

    // std::uint32_t bits = std::bit_cast<std::uint32_t>(x);

    // std::cout << std::bitset<32>(bits) << '\n';

    // // Optional: split into sign | exponent | mantissa
    // std::bitset<32> b(bits);
    // std::cout
    //     << b[31] << " | "
    //     << ((bits >> 23) & 0xFF) << " | "
    //     << (bits & 0x7FFFFF) << '\n';


    // for(int i = 0; i < 32; i++) {
    //     std::cout << "i: " << i << ", (1 << i): " << (1 << i) << std::endl;
    // }

    return 0;
}
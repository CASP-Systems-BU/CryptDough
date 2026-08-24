#include <cassert>
#include <cmath>

#include "simulated_float.h"

namespace {

using cdough::SimulatedFloat;

void TestRepresentation() {
  const SimulatedFloat value(1000, 4);
  assert(value.data() == 1000);
  assert(value.precision() == 4);
  assert(std::abs(static_cast<double>(value) - 0.1) < 1e-12);
}

void TestArithmetic() {
  const SimulatedFloat left(1000, 2);
  const SimulatedFloat right(100, 2);

  const SimulatedFloat product = left * right;
  assert(product.data() == 1000);
  assert(product.precision() == 2);

  const SimulatedFloat quotient = left / right;
  assert(quotient.data() == 1000);
  assert(quotient.precision() == 2);
}

void TestMixedPrecision() {
  const SimulatedFloat coarse(12, 1);
  const SimulatedFloat fine(125, 2);

  const SimulatedFloat sum = coarse + fine;
  assert(sum.data() == 245);
  assert(sum.precision() == 2);

  const SimulatedFloat product = coarse * fine;
  assert(product.data() == 150);
  assert(product.precision() == 2);
}

void TestRoundingAndSigns() {
  const SimulatedFloat positive(15, 1);
  const SimulatedFloat two(20, 1);
  const SimulatedFloat negative(-15, 1);

  assert((positive / two).data() == 8);
  assert((negative / two).data() == -8);
  assert(-positive == negative);
}

}  // namespace

int main() {
  TestRepresentation();
  TestArithmetic();
  TestMixedPrecision();
  TestRoundingAndSigns();
  return 0;
}
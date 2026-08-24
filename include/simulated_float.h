/**
 * @file simulated_float.h
 *
 * Decimal fixed-point arithmetic backed by signed 64-bit storage.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ostream>
#include <type_traits>

namespace cdough {

/**
 * Simulates floating-point values with decimal fixed-point arithmetic.
 *
 * A value represents `data / 10^precision`. Arithmetic results use the greater
 * operand precision. Intermediate and stored overflow is intentionally unchecked.
 */
class SimulatedFloat {
 public:
  using Storage = std::int64_t;
  using Precision = std::uint32_t;

  static constexpr Precision kDefaultPrecision = 4;

  SimulatedFloat() = default;

  template <typename Number,
            typename std::enable_if<std::is_arithmetic<Number>::value,
                                    int>::type = 0>
  SimulatedFloat(Number value)
      : data_(Quantize(static_cast<long double>(value), kDefaultPrecision)),
        precision_(kDefaultPrecision) {}

  constexpr SimulatedFloat(Storage data, Precision precision)
      : data_(data), precision_(precision) {}

  [[nodiscard]] constexpr Storage data() const { return data_; }
  [[nodiscard]] constexpr Precision precision() const { return precision_; }

  [[nodiscard]] explicit operator double() const {
    return static_cast<double>(data_) /
           static_cast<double>(Scale(precision_));
  }

  SimulatedFloat& operator+=(const SimulatedFloat& other) {
    *this = *this + other;
    return *this;
  }

  SimulatedFloat& operator-=(const SimulatedFloat& other) {
    *this = *this - other;
    return *this;
  }

  SimulatedFloat& operator*=(const SimulatedFloat& other) {
    *this = *this * other;
    return *this;
  }

  SimulatedFloat& operator/=(const SimulatedFloat& other) {
    *this = *this / other;
    return *this;
  }

  [[nodiscard]] constexpr SimulatedFloat operator-() const {
    return SimulatedFloat(-data_, precision_);
  }

  friend SimulatedFloat operator+(const SimulatedFloat& left,
                                  const SimulatedFloat& right) {
    const Precision result_precision =
        std::max(left.precision_, right.precision_);
    return SimulatedFloat(left.Rescale(result_precision) +
                              right.Rescale(result_precision),
                          result_precision);
  }

  friend SimulatedFloat operator-(const SimulatedFloat& left,
                                  const SimulatedFloat& right) {
    const Precision result_precision =
        std::max(left.precision_, right.precision_);
    return SimulatedFloat(left.Rescale(result_precision) -
                              right.Rescale(result_precision),
                          result_precision);
  }

  friend SimulatedFloat operator*(const SimulatedFloat& left,
                                  const SimulatedFloat& right) {
    const Precision result_precision =
        std::max(left.precision_, right.precision_);
    const Precision digits_to_remove =
        left.precision_ + right.precision_ - result_precision;
    return SimulatedFloat(
        DivideRounded(left.data_ * right.data_, Scale(digits_to_remove)),
        result_precision);
  }

  friend SimulatedFloat operator/(const SimulatedFloat& left,
                                  const SimulatedFloat& right) {
    const Precision result_precision =
        std::max(left.precision_, right.precision_);
    const Precision digits_to_add =
        result_precision + right.precision_ - left.precision_;
    return SimulatedFloat(
        DivideRounded(left.data_ * Scale(digits_to_add), right.data_),
        result_precision);
  }

  friend bool operator==(const SimulatedFloat& left,
                         const SimulatedFloat& right) {
    const Precision precision = std::max(left.precision_, right.precision_);
    return left.Rescale(precision) == right.Rescale(precision);
  }

  friend bool operator<(const SimulatedFloat& left,
                        const SimulatedFloat& right) {
    const Precision precision = std::max(left.precision_, right.precision_);
    return left.Rescale(precision) < right.Rescale(precision);
  }

  friend bool operator!=(const SimulatedFloat& left,
                         const SimulatedFloat& right) {
    return !(left == right);
  }

  friend bool operator>(const SimulatedFloat& left,
                        const SimulatedFloat& right) {
    return right < left;
  }

  friend bool operator<=(const SimulatedFloat& left,
                         const SimulatedFloat& right) {
    return !(right < left);
  }

  friend bool operator>=(const SimulatedFloat& left,
                         const SimulatedFloat& right) {
    return !(left < right);
  }

  friend std::ostream& operator<<(std::ostream& output,
                                  const SimulatedFloat& value) {
    return output << static_cast<double>(value);
  }

 private:
  [[nodiscard]] static constexpr Storage Scale(Precision precision) {
    Storage scale = 1;
    for (Precision digit = 0; digit < precision; ++digit) {
      scale *= 10;
    }
    return scale;
  }

  [[nodiscard]] static Storage Quantize(long double value,
                                        Precision precision) {
    return static_cast<Storage>(
        std::round(value * static_cast<long double>(Scale(precision))));
  }

  [[nodiscard]] static Storage DivideRounded(Storage numerator,
                                             Storage denominator) {
    return static_cast<Storage>(std::round(
        static_cast<long double>(numerator) /
        static_cast<long double>(denominator)));
  }

  [[nodiscard]] constexpr Storage Rescale(Precision precision) const {
    return data_ * Scale(precision - precision_);
  }

  Storage data_ = 0;
  Precision precision_ = kDefaultPrecision;
};

}  // namespace cdough
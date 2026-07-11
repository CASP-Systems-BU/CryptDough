#pragma once

#include "core/containers/matrix/hybrid/matrix.h"
#include "core/containers/matrix/hybrid/plain_matrix.h"
#include "core/containers/matrix/hybrid/secure_matrix.h"

namespace cdough::matrix {

template <typename T, template <typename> class V>
using PlainMatrix = cdough::matrix::hybrid::PlainMatrix<T, V>;

template <typename T, template <typename> class V>
using SecureMatrix = cdough::matrix::hybrid::SecureMatrix<T, V>;

using HeightWidth = cdough::matrix::hybrid::HeightWidth;

}  // namespace cdough::matrix
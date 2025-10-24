#include <algorithm>

#include "FloatMatrix.hpp"
#include "ReLU.hpp"

namespace nnn {

  void ReLU::Evaluate(FloatMatrix& input) const {
    input.MapInPlace([](float x) { return std::max(x, 0.0f); });
  }
}  // namespace nnn

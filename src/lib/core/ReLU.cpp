#include "FloatMatrix.hpp"
#include "ReLU.hpp"

namespace nnn {

  void ReLU::Evaluate(FloatMatrix& input) const {
    input.MapInPlace([](float x) { return (x < 0.0f) ? 0.0f : x; });
  }
}  // namespace nnn

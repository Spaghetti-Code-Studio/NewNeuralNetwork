#include "FloatMatrix.hpp"
#include "LeakyReLU.hpp"

nnn::LeakyReLU::LeakyReLU(float alpha) : m_alpha(alpha) {}

void nnn::LeakyReLU::Evaluate(FloatMatrix& input) const {
  const float alpha = m_alpha;
  input.MapInPlace([alpha](float x) { return x > 0 ? x : x * alpha; });
}

void nnn::LeakyReLU::Derivative(FloatMatrix& input) const {
  const float alpha = m_alpha;
  input.MapInPlace([alpha](float x) { return x > 0 ? 1 : alpha; });
}

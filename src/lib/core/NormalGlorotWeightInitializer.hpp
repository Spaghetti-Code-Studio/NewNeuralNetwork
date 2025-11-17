#pragma once

#include <random>

#include "FloatMatrix.hpp"
#include "IWeightInitializer.hpp"

namespace nnn {

  /// @brief Suitable for softmax layers 
  class NormalGlorotWeightInitializer : public IWeightInitializer {
   public:
    NormalGlorotWeightInitializer();
    NormalGlorotWeightInitializer(unsigned int seed);
    FloatMatrix Initialize(size_t n, size_t m) override;
   private:
    std::mt19937 m_rng;
  };
}  // namespace nnn
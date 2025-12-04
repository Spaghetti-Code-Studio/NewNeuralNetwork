#pragma once

#include <random>

#include "FloatMatrix.hpp"
#include "IWeightInitializer.hpp"

namespace nnn {

  /**
   * @brief Suitable for ReLU layers
   * */
  class NormalHeWeightInitializer : public IWeightInitializer {
   public:
    NormalHeWeightInitializer();
    NormalHeWeightInitializer(unsigned int seed);
    FloatMatrix Initialize(size_t row, size_t col) override;

   private:
    std::mt19937 m_rng;
  };
}  // namespace nnn
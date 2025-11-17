#pragma once

#include "FloatMatrix.hpp"

namespace nnn {

  class IWeightInitializer {
   public:
    virtual ~IWeightInitializer() = 0;

    virtual FloatMatrix Initialize(size_t row, size_t col) = 0;
  };

  inline IWeightInitializer::~IWeightInitializer() = default;
}  // namespace nnn

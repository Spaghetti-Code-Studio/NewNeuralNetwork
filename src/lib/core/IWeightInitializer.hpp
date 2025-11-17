#pragma once

#include "FloatMatrix.hpp"

namespace nnn {

  class IWeightInitializer {
   public:
    virtual ~IWeightInitializer() = 0;
    virtual FloatMatrix Initialize(size_t n, size_t m) = 0;
  };

  inline IWeightInitializer::~IWeightInitializer() = default;
}  // namespace nnn

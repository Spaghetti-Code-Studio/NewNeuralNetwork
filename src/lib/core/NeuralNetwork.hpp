#pragma once

#include <memory>
#include <vector>

#include "FloatMatrix.hpp"
#include "ILayer.hpp"

namespace nnn {
  class NeuralNetwork {
   public:
    NeuralNetwork() = default;

    size_t AddLayer(std::unique_ptr<ILayer>&& layer);
    ILayer* GetLayer(size_t index);

    FloatMatrix RunForwardPass(FloatMatrix input);

   private:
    std::vector<std::unique_ptr<ILayer>> m_layers;
  };
}  // namespace nnn

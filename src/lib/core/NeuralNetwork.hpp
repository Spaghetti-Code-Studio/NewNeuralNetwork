#pragma once

#include <memory>
#include <vector>

#include "FloatMatrix.hpp"
#include "ILayer.hpp"

namespace nnn {
  class NeuralNetwork {
   public:
    NeuralNetwork() = default;

    void AddLayer(std::unique_ptr<ILayer>&& layer);

    FloatMatrix RunForwardPass(FloatMatrix input);

    ILayer* GetLayer(size_t index);

   private:
    std::vector<std::unique_ptr<ILayer>> m_layers;
  };
}  // namespace nnn

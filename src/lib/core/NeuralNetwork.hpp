#pragma once

#include <memory>
#include <vector>

#include "FloatMatrix.hpp"
#include "ILayer.hpp"

namespace nnn {
  class NeuralNetwork {
   public:
    struct HyperParameters {
      float learningRate = 0.001f;
    };

    NeuralNetwork() = default;

    size_t AddLayer(std::unique_ptr<ILayer>&& layer);
    ILayer* GetLayer(size_t index);

    FloatMatrix RunForwardPass(FloatMatrix input);
    void Train(const FloatMatrix& input, const FloatMatrix& expected, HyperParameters params);

   private:
    std::vector<std::unique_ptr<ILayer>> m_layers;
  };
}  // namespace nnn

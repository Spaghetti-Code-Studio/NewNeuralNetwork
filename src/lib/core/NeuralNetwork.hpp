#pragma once

#include <utility>
#include <functional>
#include <memory>
#include <vector>

#include "FloatMatrix.hpp"
#include "ILayer.hpp"
#include "IOutputLayer.hpp"

namespace nnn {
  /**
   * @warning This API is not safe yet, for example, user is not anyhow forced to set output layer leading to null
   * pointer exception.
   */
  class NeuralNetwork {
   public:
    struct HyperParameters {
      float learningRate = 0.001f;
    };

    NeuralNetwork() = default;

    size_t AddHiddenLayer(std::unique_ptr<ILayer>&& layer);
    size_t SetOutputLayer(std::unique_ptr<IOutputLayer>&& layer);
    ILayer* GetLayer(size_t index);

    FloatMatrix RunForwardPass(FloatMatrix input);

    // TODO: implement this
    FloatMatrix RunBackwardPass(FloatMatrix input);

    // TODO: this will take epochs number and will do the automatic training on batches (we must take whole training
    // dataset here)
    void Train(const FloatMatrix& input, const FloatMatrix& expected, HyperParameters params);

   protected:
    std::unique_ptr<IOutputLayer> m_outputLayer;
    std::vector<std::unique_ptr<ILayer>> m_hiddenLayers;

    virtual void ForEachLayerImpl(const std::function<void(ILayer&)>& func) {
      for (auto& layer : m_hiddenLayers) {
        func(*layer);
      }
      func(*m_outputLayer);
    }

    virtual void ForEachLayerImpl(const std::function<void(const ILayer&)>& func) const {
      for (const auto& layer : m_hiddenLayers) {
        func(*layer);
      }
      func(*m_outputLayer);
    }

    template <typename Function>
    void ForEachLayer(Function&& func) {
      ForEachLayerImpl(std::function<void(ILayer&)>(std::forward<Function>(func)));
    }

    template <typename Function>
    void ForEachLayer(Function&& func) const {
      ForEachLayerImpl(std::function<void(const ILayer&)>(std::forward<Function>(func)));
    }
  };
}  // namespace nnn

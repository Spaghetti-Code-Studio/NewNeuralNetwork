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

    virtual void ForEachLayerForwardImpl(const std::function<void(ILayer&)>& func) {
      for (auto& layer : m_hiddenLayers) {
        func(*layer);
      }
      func(*m_outputLayer);
    }

    virtual void ForEachLayerForwardImpl(const std::function<void(const ILayer&)>& func) const {
      for (const auto& layer : m_hiddenLayers) {
        func(*layer);
      }
      func(*m_outputLayer);
    }

    virtual void ForEachLayerBackwardImpl(const std::function<void(ILayer&)>& func) {
      func(*m_outputLayer);
      for (auto it = m_hiddenLayers.rbegin(); it != m_hiddenLayers.rend(); ++it) {
        func(**it);
      }
    }

    virtual void ForEachLayerBackwardImpl(const std::function<void(const ILayer&)>& func) const {
      func(*m_outputLayer);
      for (auto it = m_hiddenLayers.rbegin(); it != m_hiddenLayers.rend(); ++it) {
        func(**it);
      }
    }

    template <typename Function>
    void ForEachLayerForward(Function&& func) {
      ForEachLayerForwardImpl(std::function<void(ILayer&)>(std::forward<Function>(func)));
    }

    template <typename Function>
    void ForEachLayerForward(Function&& func) const {
      ForEachLayerForwardImpl(std::function<void(const ILayer&)>(std::forward<Function>(func)));
    }

    template <typename Function>
    void ForEachLayerBackward(Function&& func) {
      ForEachLayerBackwardImpl(std::function<void(ILayer&)>(std::forward<Function>(func)));
    }

    template <typename Function>
    void ForEachLayerBackward(Function&& func) const {
      ForEachLayerBackwardImpl(std::function<void(const ILayer&)>(std::forward<Function>(func)));
    }
  };
}  // namespace nnn

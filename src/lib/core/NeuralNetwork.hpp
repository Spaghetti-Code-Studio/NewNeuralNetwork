#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <iomanip>

#include "FloatMatrix.hpp"
#include "ILayer.hpp"
#include "IOutputLayer.hpp"
#include "TrainingDataset.hpp"

namespace nnn {
  /**
   * @warning This API is not safe yet, for example, user is not anyhow forced to set output layer leading to null
   * pointer exception.
   */
  class NeuralNetwork {
   public:
    // TODO: remove constructors so we can use mdoern C++ initializator
    struct HyperParameters {
      float learningRate = 0.001f;
      float learningRateDecay = 1.0f;
      size_t epochs = 30;

      HyperParameters() = default;
      HyperParameters(float learningRate) : learningRate(learningRate) {}
      HyperParameters(float learningRate, size_t epochs) : learningRate(learningRate), epochs(epochs) {}
      HyperParameters(float learningRate, float learningRateDecay, size_t epochs)
          : learningRate(learningRate), learningRateDecay(learningRateDecay), epochs(epochs) {}
    };

    NeuralNetwork() = default;
    NeuralNetwork(HyperParameters params) : m_params(params) {}

    size_t AddHiddenLayer(std::unique_ptr<ILayer>&& layer);
    size_t SetOutputLayer(std::unique_ptr<IOutputLayer>&& layer);
    ILayer* GetLayer(size_t index);

    FloatMatrix RunForwardPass(FloatMatrix input);
    void RunBackwardPass(FloatMatrix gradient);
    void UpdateWeights();

    struct Statistics {
      std::vector<float> trainingLosses;
      std::vector<float> validationLosses;

      void Print(int stride = 0) const {  //

        const int precision = 5;
        std::cout << std::fixed << std::setprecision(precision);

        for (size_t i = 0; i < trainingLosses.size(); i++) {
          if (stride == 0 || i % stride == 0) {
            std::cout << "Epoch <" << i << "> - training loss: <" << trainingLosses[i] << ">, validation loss: <"
                      << validationLosses[i] << ">.\n";
          }
        }

        std::cout << std::defaultfloat;
      }
    };

    Statistics Train(TrainingDataset& trainingDataset, bool reportProgress = false);

   protected:
    std::unique_ptr<IOutputLayer> m_outputLayer;
    std::vector<std::unique_ptr<ILayer>> m_hiddenLayers;
    HyperParameters m_params = HyperParameters();

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

#pragma once

#include <functional>
#include <stdexcept>

#include "NeuralNetwork.hpp"

/**
 * @brief This class extends NeuralNetwork class. It is possible to override few methods from the base class to enable
 * more profilic testing.
 */
class TestableNeuralNetwork : public nnn::NeuralNetwork {
 public:
  struct Parameters {
    bool ignoreOutputLayer = false;
  };

  TestableNeuralNetwork(Parameters params) : m_params(params) {}

 protected:
  void ForEachLayerImpl(const std::function<void(nnn::ILayer&)>& func) override {
    for (auto& layer : m_hiddenLayers) {
      func(*layer);
    }

    if (!m_params.ignoreOutputLayer) {
      if (!m_outputLayer) {
        throw std::runtime_error("Output layer is nullptr, but m_params.ignoreOutputLayer is set to <false>!");
      }
      func(*m_outputLayer);
    }
  }

  void ForEachLayerImpl(const std::function<void(const nnn::ILayer&)>& func) const override {
    for (const auto& layer : m_hiddenLayers) {
      func(*layer);
    }

    if (!m_params.ignoreOutputLayer) {
      if (!m_outputLayer) {
        throw std::runtime_error("Output layer is nullptr, but m_params.ignoreOutputLayer is set to <false>!");
      }
      func(*m_outputLayer);
    }
  }

 private:
  Parameters m_params;
};
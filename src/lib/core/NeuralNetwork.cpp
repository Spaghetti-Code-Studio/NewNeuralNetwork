#include "NeuralNetwork.hpp"
#include <cmath>
#include <assert.h>
#include <iostream>

#include "TestDataSoftmaxEvaluator.hpp"

static float ComputeCrossEntropyLoss(const nnn::FloatMatrix& predictions, const nnn::FloatMatrix& labels) {
  nnn::FloatMatrix logPredictions = predictions.Map([](float x) { 
      return std::log(std::max(1e-10f, std::min(1.0f - 1e-10f, x))); 
  });

  auto loss = labels.Hadamard(logPredictions);
  loss.Transpose();
  
  auto flat = nnn::FloatMatrix::SumColumns(loss);
  flat.Transpose();
  auto total = nnn::FloatMatrix::SumColumns(flat);
  
  return -total(0, 0) / loss.GetRowCount();
}

namespace nnn {

  size_t NeuralNetwork::AddHiddenLayer(std::unique_ptr<ILayer>&& layer) {
    m_hiddenLayers.push_back(std::move(layer));
    return m_hiddenLayers.size() - 1;
  }

  size_t NeuralNetwork::SetOutputLayer(std::unique_ptr<IOutputLayer>&& layer) {
    m_outputLayer = std::move(layer);
    return m_hiddenLayers.size();
  }

  FloatMatrix NeuralNetwork::RunForwardPass(FloatMatrix input) {
    ForEachLayerForward([&](ILayer& layer) { input = layer.Forward(input); });
    return input;
  }

  void NeuralNetwork::RunBackwardPass(FloatMatrix gradient) {
    ForEachLayerBackward([&](ILayer& layer) { gradient = layer.Backward(gradient); });
  }

  void NeuralNetwork::UpdateWeights() {
    ForEachLayerForward([&](ILayer& layer) {
      FloatMatrix& weightVelocity = layer.GetWightsVelocity();
      FloatMatrix& biasVelocity = layer.GetBiasesVelocity();
      
      weightVelocity = weightVelocity * m_params.momentum + layer.GetWightsGradient();
      biasVelocity = biasVelocity * m_params.momentum + layer.GetBiasesGradient();
    
      FloatMatrix newWeights = layer.GetWeights() * (1 - m_params.learningRate * m_params.weightDecay)
                               - weightVelocity * m_params.learningRate;
      FloatMatrix newBiases = layer.GetBiases() - biasVelocity * m_params.learningRate;

      layer.Update(newWeights, newBiases);
    });
  }

  NeuralNetwork::Statistics NeuralNetwork::Train(TrainingDataset& trainingDataset, bool reportProgress) {  //

    auto lossesValidation = std::vector<float>();
    auto lossesTraining = std::vector<float>();
    lossesValidation.reserve(m_params.epochs);
    lossesTraining.reserve(m_params.epochs);

    FloatMatrix allTrainFeatures = trainingDataset.GetTrainingFeatures();
    FloatMatrix allTrainLabels = trainingDataset.GetTrainingLabels();
    FloatMatrix allValidationFeatures = trainingDataset.GetValidationFeatures();
    FloatMatrix allValidationLabels = trainingDataset.GetValidationLabels();

    for (size_t epoch = 0; epoch < m_params.epochs; ++epoch) {
      while (trainingDataset.HasNextBatch()) {
        auto input = trainingDataset.GetNextBatch();
        FloatMatrix actual = RunForwardPass(input.features);
        FloatMatrix gradient = m_outputLayer->ComputeOutputGradient(actual, input.labels);
        gradient.MapInPlace([input](float x) { return x / input.features.GetColCount(); });
        RunBackwardPass(gradient);
        UpdateWeights();
      }
      trainingDataset.Reset();

      FloatMatrix trainPredictions = RunForwardPass(allTrainFeatures);
      float trainLoss = ComputeCrossEntropyLoss(trainPredictions, allTrainLabels);
      
      lossesTraining.push_back(trainLoss);

      if (trainingDataset.HasValidationDataset() && reportProgress) {
        FloatMatrix actual = RunForwardPass(allValidationFeatures);
        float validationLoss = ComputeCrossEntropyLoss(actual, allValidationLabels);

        auto validationEval = TestDataSoftmaxEvaluator::Evaluate(actual, allValidationLabels);
        float percentValidation = static_cast<float>(validationEval.correctlyClassifiedCount) / validationEval.totalExamplesCount;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Epoch " << epoch + 1 << "/" << m_params.epochs << " - training loss: " << trainLoss;
        std::cout << ", validation loss: " << validationLoss;
        std::cout << std::setprecision(2) << " (aprox. " << percentValidation * 100 << "%)";
        std::cout << "." << std::endl;
      }

      m_params.learningRate *= m_params.learningRateDecay;
    }

    return {lossesTraining, lossesValidation};
  }

  ILayer* NeuralNetwork::GetLayer(size_t index) {  //

    if (index > m_hiddenLayers.size()) {
      return nullptr;
    }

    if (index == m_hiddenLayers.size()) {
      return m_outputLayer.get();
    }

    return m_hiddenLayers[index].get();
  }
}  // namespace nnn

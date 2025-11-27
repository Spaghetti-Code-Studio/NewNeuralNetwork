#include <chrono>
#include <iostream>
#include <memory>

#include <Config.hpp>
#include <CSVReader.hpp>
#include <DataLoader.hpp>
#include <DenseLayer.hpp>
#include <LeakyReLU.hpp>
#include <NeuralNetwork.hpp>
#include <NormalGlorotWeightInitializer.hpp>
#include <NormalHeWeightInitializer.hpp>
#include <SoftmaxDenseOutputLayer.hpp>
#include <TestDataSoftmaxEvaluator.hpp>
#include <Timer.hpp>

int main(int argc, char* argv[]) {  //

  nnn::Config config;
  auto configResult = config.LoadFromJSON("../../../../config.json");
  if (configResult.has_error()) {
    std::cout << configResult.error() << std::endl;
    return -1;
  }

  std::cout << "--- NewNeuralNetwork ---\n"
            << "Training neural network on MNIST fashion dataset.\n"
            << std::endl;

  std::cout << config.ToString() << std::endl;

  int seed = config.randomSeed;
  auto glorotInit = nnn::NormalGlorotWeightInitializer(seed);
  auto heInit = nnn::NormalHeWeightInitializer(seed);

  auto neuralNetwork = nnn::NeuralNetwork(nnn::NeuralNetwork::HyperParameters(
      config.learningRate, config.epochs));  // TODO: use modern C++ initializator

  if (config.layers.size() == 0) {
    std::cout << "No layers were defined. Neural network cannot be constructed!" << std::endl;
    return -1;
  }

  for (int layerIndex = 0; layerIndex < config.layers.size() - 1; ++layerIndex) {
    neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(config.layers[layerIndex].inputNumber,
        config.layers[layerIndex].outputNumber, std::make_unique<nnn::LeakyReLU>(), heInit));
  }
  neuralNetwork.SetOutputLayer(
      std::make_unique<nnn::SoftmaxDenseOutputLayer>(config.layers[config.layers.size() - 1].inputNumber,
          config.layers[config.layers.size() - 1].outputNumber, glorotInit));

  nnn::Timer timer;
  timer.Start();
  std::cout << "Loading dataset..." << std::endl;
  auto reader = std::make_shared<nnn::CSVReader>();
  auto datasetResult = nnn::DataLoader::Load({.trainingFeatures = "../../../../data/fashion_mnist_train_vectors.csv",
                                                 .trainingLabels = "../../../../data/fashion_mnist_train_labels.csv",
                                                 .testingFeatures = "../../../../data/fashion_mnist_test_vectors.csv",
                                                 .testingLabels = "../../../../data/fashion_mnist_test_labels.csv"},
      reader, {.batchSize = config.batchSize, .validationSetFraction = config.validationSetFraction},
      {.expectedClassNumber = config.expectedClassNumber, .shouldOneHotEncode = true, .normalizationFactor = 256});

  if (datasetResult.has_error()) {
    std::cout << datasetResult.error() << std::endl;
    return -1;
  }
  std::cout << "Loading of dataset took " << timer.End() << " seconds.\n" << std::endl;

  std::cout << "Training neural network..." << std::endl;
  timer.Start();
  auto dataset = datasetResult.value();
  auto statistics = neuralNetwork.Train(dataset.trainingDataset, true);
  std::cout << "Training took " << timer.End() << " seconds." << std::endl;

  std::cout << "\nEvaluation of neural network on testing data..." << std::endl;
  auto result = neuralNetwork.RunForwardPass(*dataset.testingFeatures);
  auto evaluation = nnn::TestDataSoftmaxEvaluator::Evaluate(result, *dataset.testingLabels);
  evaluation.Print();

  return 0;
}

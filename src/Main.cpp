#include <chrono>
#include <iostream>
#include <memory>

#include <CSVReader.hpp>
#include <DataLoader.hpp>
#include <DenseLayer.hpp>
#include <FloatMatrix.hpp>
#include <LeakyReLU.hpp>
#include <NeuralNetwork.hpp>
#include <NormalGlorotWeightInitializer.hpp>
#include <SoftmaxDenseOutputLayer.hpp>
#include <TestDataSoftmaxEvaluator.hpp>
#include <Timer.hpp>
#include <NormalHeWeightInitializer.hpp>

int main(int argc, char* argv[]) {  //

  std::cout << "Starting" << std::endl;

  nnn::Timer timer;

  int seed = 42;
  auto glorotInit = nnn::NormalGlorotWeightInitializer(seed);
  auto heInit = nnn::NormalHeWeightInitializer(seed);

  auto neuralNetwork = nnn::NeuralNetwork(nnn::NeuralNetwork::HyperParameters(0.01f, 5));  // learning rate, epochs

  size_t l1 = neuralNetwork.AddHiddenLayer(
      std::make_unique<nnn::DenseLayer>(784, 128, std::make_unique<nnn::LeakyReLU>(0.05f), heInit));
  size_t l2 = neuralNetwork.AddHiddenLayer(
      std::make_unique<nnn::DenseLayer>(128, 54, std::make_unique<nnn::LeakyReLU>(0.05f), heInit));
  size_t l3 = neuralNetwork.SetOutputLayer(std::make_unique<nnn::SoftmaxDenseOutputLayer>(54, 10, glorotInit));

  std::cout << "Loading data..." << std::endl;
  auto reader = std::make_shared<nnn::CSVReader>();
  auto datasetResult = nnn::DataLoader::Load({.trainingFeatures = "../../../../data/fashion_mnist_train_vectors.csv",
                                                 .trainingLabels = "../../../../data/fashion_mnist_train_labels.csv",
                                                 .testingFeatures = "../../../../data/fashion_mnist_test_vectors.csv",
                                                 .testingLabels = "../../../../data/fashion_mnist_test_labels.csv"},
      reader, {.batchSize = 256, .validationSetFraction = 0.2f},
      {.expectedClassNumber = 10, .shouldOneHotEncode = true, .normalizationFactor = 256});

  if (datasetResult.has_error()) {
    std::cout << datasetResult.error() << std::endl;
    return -1;
  }

  std::cout << "Training data..." << std::endl;
  timer.Start();
  auto statistics = neuralNetwork.Train(datasetResult.value().trainingDataset);
  std::cout << "Training time: " << timer.End() << " seconds.\n" << std::endl;
  statistics.Print();

  std::cout << "Evaluation..." << std::endl;
  auto result = neuralNetwork.RunForwardPass(*datasetResult.value().testingFeatures);
  auto evaluation = nnn::TestDataSoftmaxEvaluator::Evaluate(result, *datasetResult.value().testingLabels);
  evaluation.Print();

  return 0;
}

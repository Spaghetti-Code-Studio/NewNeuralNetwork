#include <chrono>
#include <iostream>
#include <memory>

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

  std::cout << "NewNeuralNetwork\n"
            << "Training neural network on MNIST fashion dataset.\n"
            << std::endl;

  int seed = 42;
  auto glorotInit = nnn::NormalGlorotWeightInitializer(seed);
  auto heInit = nnn::NormalHeWeightInitializer(seed);

  auto neuralNetwork = nnn::NeuralNetwork(nnn::NeuralNetwork::HyperParameters(0.01f, 5));  // learning rate, epochs

  size_t l1 = neuralNetwork.AddHiddenLayer(
      std::make_unique<nnn::DenseLayer>(784, 128, std::make_unique<nnn::LeakyReLU>(0.05f), heInit));
  size_t l2 = neuralNetwork.AddHiddenLayer(
      std::make_unique<nnn::DenseLayer>(128, 54, std::make_unique<nnn::LeakyReLU>(0.05f), heInit));
  size_t l3 = neuralNetwork.SetOutputLayer(std::make_unique<nnn::SoftmaxDenseOutputLayer>(54, 10, glorotInit));

  nnn::Timer timer;
  timer.Start();
  std::cout << "Loading dataset..." << std::endl;
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
  } else {
    std::cout << "Loading of dataset took " << timer.End() << " seconds.\n" << std::endl;
  }

  std::cout << "Training neural network..." << std::endl;
  timer.Start();
  auto dataset = datasetResult.value();
  auto statistics = neuralNetwork.Train(dataset.trainingDataset);
  std::cout << "Training took " << timer.End() << " seconds." << std::endl;
  statistics.Print();

  std::cout << "\nEvaluation of neural network on testing data..." << std::endl;
  auto result = neuralNetwork.RunForwardPass(*dataset.testingFeatures);
  auto evaluation = nnn::TestDataSoftmaxEvaluator::Evaluate(result, *dataset.testingLabels);
  evaluation.Print();

  return 0;
}

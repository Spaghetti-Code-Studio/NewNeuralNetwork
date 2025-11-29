#include <chrono>
#include <iostream>
#include <memory>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

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


// https://patorjk.com/software/taag/#p=display&f=Small&t=NewNeuralNetwork&x=none&v=4&h=4&w=80&we=false
const std::string Logo = R"(
 _  _            _  _                   _ _  _     _                  _   
| \| |_____ __ _| \| |___ _  _ _ _ __ _| | \| |___| |___ __ _____ _ _| |__
| .` / -_) V  V / .` / -_) || | '_/ _` | | .` / -_)  _\ V  V / _ \ '_| / /
|_|\_\___|\_/\_/|_|\_\___|\_,_|_| \__,_|_|_|\_\___|\__|\_/\_/\___/_| |_\_\                                                                           
)";

int main(int argc, char* argv[]) {  //

  nnn::Config config;
  auto configResult = config.LoadFromJSON("../../../../config.json");
  if (configResult.has_error()) {
    std::cout << configResult.error() << std::endl;
    return -1;
  }

  std::cout << Logo << "\nVersion 1.0.0\n"
            << "Training neural network on MNIST fashion dataset.\n"
            << std::endl;

  std::cout << config.ToString() << std::endl;

#ifdef _OPENMP
  omp_set_num_threads(config.hardThreadsLimit);
  std::cout << "Parallel computing on.\n" << std::endl;
#endif

  int seed = config.randomSeed;
  auto glorotInit = nnn::NormalGlorotWeightInitializer(seed);
  auto heInit = nnn::NormalHeWeightInitializer(seed);

  auto neuralNetwork = nnn::NeuralNetwork(nnn::NeuralNetwork::HyperParameters(
      config.learningRate, config.learningRateDecay, config.weightDecay, config.momentum, config.epochs));  // TODO: use modern C++ initializator

  if (config.layers.size() < 2) {
    std::cout << "At least two layers are required. Neural network cannot be constructed!" << std::endl;
    return -1;
  }

  for (int i = 0; i < config.layers.size() - 2; ++i) {
    neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(
        config.layers[i], config.layers[i + 1], std::make_unique<nnn::LeakyReLU>(), heInit));
  }

  neuralNetwork.SetOutputLayer(std::make_unique<nnn::SoftmaxDenseOutputLayer>(
      config.layers[config.layers.size() - 2], config.layers[config.layers.size() - 1], glorotInit));

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
  neuralNetwork.Train(dataset.trainingDataset, true);  // ignore statistics output - print it live
  std::cout << "Training took " << timer.End() << " seconds." << std::endl;

  std::cout << "\nEvaluation of neural network on testing data..." << std::endl;
  auto result = neuralNetwork.RunForwardPass(*dataset.testingFeatures);
  auto evaluation = nnn::TestDataSoftmaxEvaluator::Evaluate(result, *dataset.testingLabels);
  evaluation.Print();

  return 0;
}

#include <chrono>
#include <iostream>
#include <memory>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Config.hpp>
#include <CSVLabelWriter.hpp>
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

/**
 * @brief Logo for the program. Source:
 * https://patorjk.com/software/taag/#p=display&f=Small&t=NewNeuralNetwork&x=none&v=4&h=4&w=80&we=false.
 */
const std::string Logo = R"(
 _  _            _  _                   _ _  _     _                  _   
| \| |_____ __ _| \| |___ _  _ _ _ __ _| | \| |___| |___ __ _____ _ _| |__
| .` / -_) V  V / .` / -_) || | '_/ _` | | .` / -_)  _\ V  V / _ \ '_| / /
|_|\_\___|\_/\_/|_|\_\___|\_,_|_| \__,_|_|_|\_\___|\__|\_/\_/\___/_| |_\_\)";

int main(int argc, char* argv[]) {  //

  nnn::Config config;

#ifdef IS_PRODUCTION_BUILD
  const std::string PREFIX = "./";
#else
  const std::string PREFIX = "../../../../";
#endif

  auto configResult = config.LoadFromJSON(PREFIX + "config.json");

  if (configResult.has_error()) {
    std::cout << configResult.error() << std::endl;
    return -1;
  }

  std::cout << Logo << "\n\nVersion 1.0.0\n"
            << "Training neural network on MNIST fashion dataset.\n"
            << std::endl;
  std::cout << config.ToString() << std::endl;

#ifdef _OPENMP
  omp_set_num_threads(config.hardThreadsLimit);
  std::cout << "Parallel computing is enabled.\n" << std::endl;
#else
  std::cout << "Parallel computing is not enabled (missing OpenMP dependency).\n" << std::endl;
#endif

  int seed = config.randomSeed;
  auto glorotInit = nnn::NormalGlorotWeightInitializer(seed);
  auto heInit = nnn::NormalHeWeightInitializer(seed);

  auto neuralNetwork = nnn::NeuralNetwork({.learningRate = config.learningRate,
      .learningRateDecay = config.learningRateDecay,
      .weightDecay = config.weightDecay,
      .momentum = config.momentum,
      .epochs = config.epochs});

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

  auto datasetResult = nnn::DataLoader::Load({.trainingFeatures = PREFIX + "data/fashion_mnist_train_vectors.csv",
                                                 .trainingLabels = PREFIX + "data/fashion_mnist_train_labels.csv",
                                                 .testingFeatures = PREFIX + "data/fashion_mnist_test_vectors.csv",
                                                 .testingLabels = PREFIX + "data/fashion_mnist_test_labels.csv"},
      reader, {.batchSize = config.batchSize, .validationSetFraction = config.validationSetFraction},
      {.expectedClassNumber = config.expectedClassNumber, .shouldOneHotEncode = true, .normalizationFactor = 256});

  if (datasetResult.has_error()) {
    std::cout << datasetResult.error() << std::endl;
    return -1;
  }
  std::cout << "Loading of dataset took " << timer.End() << " seconds.\n" << std::endl;

  timer.Start();
  std::cout << "Training neural network..." << std::endl;
  auto dataset = datasetResult.value();
  neuralNetwork.Train(dataset.trainingDataset, true);
  std::cout << "Training took " << timer.End() << " seconds." << std::endl;

  timer.Start();
  std::cout << "\nEvaluation of neural network on testing data..." << std::endl;

  auto testEval = neuralNetwork.RunForwardPass(*dataset.testingFeatures);
  auto trainEval = neuralNetwork.RunForwardPass(*dataset.trainingDataset.GetFeatures());

  auto evaluation = nnn::TestDataSoftmaxEvaluator::Evaluate(testEval, *dataset.testingLabels);
  evaluation.Print();

  std::cout << "Evaluation took " << timer.End() << " seconds." << std::endl;

  timer.Start();
  std::cout << "\nWriting results into CSV files..." << std::endl;
  nnn::CSVLabelsWriter writer;

  auto writeResultTest = writer.Write(PREFIX + "test_predictions.csv", testEval);
  auto writeResultTrain = writer.Write(PREFIX + "train_predictions.csv", trainEval);

  if (writeResultTest.has_error()) {
    std::cout << writeResultTest.error() << std::endl;
    return -1;
  }
  if (writeResultTrain.has_error()) {
    std::cout << writeResultTrain.error() << std::endl;
    return -1;
  }
  std::cout << "Writing results took " << timer.End() << " seconds." << std::endl;

  return 0;
}

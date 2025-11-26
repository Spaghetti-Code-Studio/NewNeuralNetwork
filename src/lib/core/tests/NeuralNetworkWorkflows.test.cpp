#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <memory>

#include "CrossEntropyWithSoftmax.hpp"
#include "CSVReader.hpp"
#include "DataLoader.hpp"
#include "DenseLayer.hpp"
#include "FloatMatrix.hpp"
#include "ILayer.hpp"
#include "LeakyReLU.hpp"
#include "NeuralNetwork.hpp"
#include "NormalGlorotWeightInitializer.hpp"
#include "NormalHeWeightInitializer.hpp"
#include "ReLU.hpp"
#include "Softmax.hpp"
#include "SoftmaxDenseOutputLayer.hpp"
#include "TrainingDataset.hpp"

#include "TestableNeuralNetwork.hpp"

// ------------------------------------------------------------------------------------------------

/**
 * @brief Retrieves given layer from the neural network.
 *
 * @tparam LayerType type of the layer to retrieve
 * @param neuralNetwork neural network
 * @param index index of the layer in the network
 * @returns pointer to the layer or null pointer if the index is wrong or the type does not match
 */
template <typename LayerType>
static inline LayerType* GetLayerAs(nnn::NeuralNetwork* neuralNetwork, size_t index) {
  nnn::ILayer* baseLayer = neuralNetwork->GetLayer(index);
  return dynamic_cast<LayerType*>(baseLayer);
}

// // ------------------------------------------------------------------------------------------------

TEST_CASE("1 Layer NN - Basic forward pass with ReLU") {  //

  auto neuralNetwork = TestableNeuralNetwork({true});
  size_t index = neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(3, 2, std::make_unique<nnn::ReLU>()));

  auto newBiases = nnn::FloatMatrix::Create(2, 1, {-5.0f, 0.5f}).value();
  auto newWeights = nnn::FloatMatrix::Create(2, 3, {1.0f, 2.0f, 0.0f, 0.5f, 0.5f, 1.0f}).value();

  nnn::DenseLayer* layer = GetLayerAs<nnn::DenseLayer>(&neuralNetwork, index);
  layer->Update(newWeights, newBiases);

  nnn::FloatMatrix input = nnn::FloatMatrix::Create(3, 1, {1.0f, 2.0f, 3.0f}).value();
  auto result = neuralNetwork.RunForwardPass(input);

  CHECK_THAT(result(0, 0), Catch::Matchers::WithinAbs(0.0f, 0.001));
  CHECK_THAT(result(1, 0), Catch::Matchers::WithinAbs(5.0f, 0.001));
}

TEST_CASE("2 Layer NN - Basic forward pass with ReLU and SoftMax") {  //
  auto neuralNetwork = nnn::NeuralNetwork();
  size_t l1 = neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(3, 3, std::make_unique<nnn::ReLU>()));
  size_t l2 = neuralNetwork.SetOutputLayer(std::make_unique<nnn::SoftmaxDenseOutputLayer>(3, 3));

  auto l1Weights = nnn::FloatMatrix::Create(3, 3, {1.0f, 2.0f, 0.0f, 0.5f, 0.5f, 1.0f, 2.0f, 1.0f, 0.0f}).value();
  auto l1Biases = nnn::FloatMatrix::Create(3, 1, {0.0f, 0.5f, 1.0f}).value();

  nnn::DenseLayer* layer1 = GetLayerAs<nnn::DenseLayer>(&neuralNetwork, l1);
  layer1->Update(l1Weights, l1Biases);

  auto l2Weights = nnn::FloatMatrix::Identity(3);
  auto l2Biases = nnn::FloatMatrix::Create(3, 1, {0.0f, 1.0f, 0.0f}).value();

  nnn::SoftmaxDenseOutputLayer* layer2 = GetLayerAs<nnn::SoftmaxDenseOutputLayer>(&neuralNetwork, l2);
  layer2->Update(l2Weights, l2Biases);

  nnn::FloatMatrix input = nnn::FloatMatrix::Create(3, 1, {1.0f, 2.0f, 3.0f}).value();
  auto result = neuralNetwork.RunForwardPass(input);

  CHECK(result.GetColCount() == 1);
  CHECK(result.GetRowCount() == 3);
  CHECK_THAT(result(0, 0), Catch::Matchers::WithinAbs(0.211942f, 0.001));
  CHECK_THAT(result(1, 0), Catch::Matchers::WithinAbs(0.576117f, 0.001));
  CHECK_THAT(result(2, 0), Catch::Matchers::WithinAbs(0.211942f, 0.001));
}

TEST_CASE("SoftMax Evaluation") {  //

  auto matrix = nnn::FloatMatrix::Create(
      5, 2, {-0.586459f, 3000.0f, 10.0f, 3000.0f, 0.0001f, 3000.0f, 8.0f, 3000.0f, -1.0f, 3000.1f})
                    .value();
  nnn::Softmax softmax;
  softmax.Evaluate(matrix);

  CHECK_THAT(matrix(0, 0), Catch::Matchers::WithinAbs(0.0000222434f, 0.001));
  CHECK_THAT(matrix(1, 0), Catch::Matchers::WithinAbs(0.8807293076f, 0.001));
  CHECK_THAT(matrix(2, 0), Catch::Matchers::WithinAbs(0.000039989f, 0.001));
  CHECK_THAT(matrix(3, 0), Catch::Matchers::WithinAbs(0.1191937503f, 0.001));
  CHECK_THAT(matrix(4, 0), Catch::Matchers::WithinAbs(0.0000147097f, 0.001));

  CHECK_THAT(matrix(0, 1), Catch::Matchers::WithinAbs(0.1958798277f, 0.001));
  CHECK_THAT(matrix(1, 1), Catch::Matchers::WithinAbs(0.1958798277f, 0.001));
  CHECK_THAT(matrix(2, 1), Catch::Matchers::WithinAbs(0.1958798277f, 0.001));
  CHECK_THAT(matrix(3, 1), Catch::Matchers::WithinAbs(0.1958798277f, 0.001));
  CHECK_THAT(matrix(4, 1), Catch::Matchers::WithinAbs(0.2164806891f, 0.001));
}

TEST_CASE("SoftMax Evaluation and CrossEntropy Loss") {  //

  auto matrix = nnn::FloatMatrix::Create(
      5, 2, {-0.586459f, 3000.0f, 10.0f, 3000.0f, 0.0001f, 3000.0f, 8.0f, 3000.0f, -1.0f, 3000.1f})
                    .value();
  nnn::Softmax softmax;
  softmax.Evaluate(matrix);

  auto expected =
      nnn::FloatMatrix::Create(5, 2, {1.0f, 1.0f, 10.0f, 10.0f, -1.0f, 10.0f, 0.0f, 0.0f, -1.0f, 2.333f}).value();
  nnn::CrossEntropyWithSoftmax crossEntropy;
  auto gradient = crossEntropy.Loss(matrix, expected);

  CHECK(gradient.GetRowCount() == 5);
  CHECK(gradient.GetColCount() == 2);
  CHECK_THAT(gradient(0, 0), Catch::Matchers::WithinAbs((0.0000222434f - 1.0f), 0.001));
  CHECK_THAT(gradient(1, 0), Catch::Matchers::WithinAbs((0.8807293076f - 10.0f), 0.001));
  CHECK_THAT(gradient(2, 0), Catch::Matchers::WithinAbs((0.000039989f + 1.0f), 0.001));
  CHECK_THAT(gradient(3, 0), Catch::Matchers::WithinAbs((0.1191937503f - 0.0f), 0.001));
  CHECK_THAT(gradient(4, 0), Catch::Matchers::WithinAbs((0.0000147097f + 1.0f), 0.001));

  CHECK_THAT(gradient(0, 1), Catch::Matchers::WithinAbs((0.1958798277f - 1.0f), 0.001));
  CHECK_THAT(gradient(1, 1), Catch::Matchers::WithinAbs((0.1958798277f - 10.0f), 0.001));
  CHECK_THAT(gradient(2, 1), Catch::Matchers::WithinAbs((0.1958798277f - 10.0f), 0.001));
  CHECK_THAT(gradient(3, 1), Catch::Matchers::WithinAbs((0.1958798277f - 0.0f), 0.001));
  CHECK_THAT(gradient(4, 1), Catch::Matchers::WithinAbs((0.2164806891f - 2.333f), 0.001));
}

TEST_CASE("2 Layer NN - Forward pass XOR with ReLU") {  //
  auto neuralNetwork = TestableNeuralNetwork({true});

  size_t l1 = neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(2, 2, std::make_unique<nnn::ReLU>()));
  size_t l2 = neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(2, 1, std::make_unique<nnn::ReLU>()));

  auto weights1 = nnn::FloatMatrix::Create(2, 2, {1.0f, 1.0f, 1.0f, 1.0f}).value();
  auto biases1 = nnn::FloatMatrix::Create(2, 1, {-1.5f, -0.5f}).value();

  nnn::DenseLayer* denseLayer1 = GetLayerAs<nnn::DenseLayer>(&neuralNetwork, l1);
  denseLayer1->Update(weights1, biases1);

  auto weights2 = nnn::FloatMatrix::Create(1, 2, {-6.0f, 2.0f}).value();
  auto biases2 = nnn::FloatMatrix::Create(1, 1, {0.0f}).value();

  nnn::DenseLayer* denseLayer2 = GetLayerAs<nnn::DenseLayer>(&neuralNetwork, l2);
  denseLayer2->Update(weights2, biases2);

  // 0 XOR 0 = 0
  auto input = nnn::FloatMatrix::Create(2, 1, {0.0f, 0.0f}).value();
  auto result = neuralNetwork.RunForwardPass(input);
  CHECK_THAT(result(0, 0), Catch::Matchers::WithinAbs(0.0f, 0.001));

  // 0 XOR 1 = 1
  input = nnn::FloatMatrix::Create(2, 1, {0.0f, 1.0f}).value();
  result = neuralNetwork.RunForwardPass(input);
  CHECK_THAT(result(0, 0), Catch::Matchers::WithinAbs(1.0f, 0.001));

  // 1 XOR 0 = 1
  input = nnn::FloatMatrix::Create(2, 1, {1.0f, 0.0f}).value();
  result = neuralNetwork.RunForwardPass(input);
  CHECK_THAT(result(0, 0), Catch::Matchers::WithinAbs(1.0f, 0.001));

  // 1 XOR 1 = 0
  input = nnn::FloatMatrix::Create(2, 1, {1.0f, 1.0f}).value();
  result = neuralNetwork.RunForwardPass(input);
  CHECK_THAT(result(0, 0), Catch::Matchers::WithinAbs(0.0f, 0.001));
}

TEST_CASE("2 Layer NN - Batch forward pass XOR with ReLU") {
  auto neuralNetwork = TestableNeuralNetwork({true});

  size_t l1 = neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(2, 2, std::make_unique<nnn::ReLU>()));
  size_t l2 = neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(2, 1, std::make_unique<nnn::ReLU>()));

  auto weights1 = nnn::FloatMatrix::Ones(2, 2);
  auto biases1 = nnn::FloatMatrix::Create(1, 2, {-1.5f, -0.5f}).value();
  biases1.Transpose();  // using transposed should act as a column vector

  nnn::DenseLayer* denseLayer1 = GetLayerAs<nnn::DenseLayer>(&neuralNetwork, l1);
  denseLayer1->Update(weights1, biases1);

  auto weights2 = nnn::FloatMatrix::Create(1, 2, {-6.0f, 2.0f}).value();
  auto biases2 = nnn::FloatMatrix::Zeroes(1, 1);

  nnn::DenseLayer* denseLayer2 = GetLayerAs<nnn::DenseLayer>(&neuralNetwork, l2);
  denseLayer2->Update(weights2, biases2);

  // compute all combinations at once
  auto input = nnn::FloatMatrix::Create(2, 4,
      {
          0.0f,
          0.0f,
          1.0f,
          1.0f,
          0.0f,
          1.0f,
          0.0f,
          1.0f,
      })
                   .value();

  auto result = neuralNetwork.RunForwardPass(input);

  CHECK(result.GetColCount() == 4);  // matrix should contain all outputs

  CHECK_THAT(result(0, 0), Catch::Matchers::WithinAbs(0.0f, 0.001));
  CHECK_THAT(result(0, 1), Catch::Matchers::WithinAbs(1.0f, 0.001));
  CHECK_THAT(result(0, 2), Catch::Matchers::WithinAbs(1.0f, 0.001));
  CHECK_THAT(result(0, 3), Catch::Matchers::WithinAbs(0.0f, 0.001));
}

TEST_CASE("2 Layer NN - Backward pass test") {  //
  auto neuralNetwork = TestableNeuralNetwork();

  size_t l1 = neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(3, 3, std::make_unique<nnn::ReLU>()));
  size_t l2 = neuralNetwork.SetOutputLayer(std::make_unique<nnn::SoftmaxDenseOutputLayer>(3, 3));

  nnn::DenseLayer* layer1 = GetLayerAs<nnn::DenseLayer>(&neuralNetwork, l1);
  nnn::SoftmaxDenseOutputLayer* layer2 = GetLayerAs<nnn::SoftmaxDenseOutputLayer>(&neuralNetwork, l2);

  size_t index = 0;
  neuralNetwork.ForEachLayerBackward([&](nnn::ILayer& layer) {
    if (index == 0) {
      CHECK(&layer == static_cast<nnn::ILayer*>(layer2));
    } else if (index == 1) {
      CHECK(&layer == static_cast<nnn::ILayer*>(layer1));
    }
    index++;
  });

  CHECK(index == 2);
}

TEST_CASE("2 Layer NN - Basic forward and backward pass with ReLU and SoftMax") {  //

  auto neuralNetwork = nnn::NeuralNetwork(nnn::NeuralNetwork::HyperParameters(0.008f));
  size_t l1 = neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(3, 3, std::make_unique<nnn::ReLU>()));
  size_t l2 = neuralNetwork.SetOutputLayer(std::make_unique<nnn::SoftmaxDenseOutputLayer>(3, 3));

  auto l1Weights = nnn::FloatMatrix::Create(3, 3, {1.0f, 2.0f, 0.0f, 0.5f, 0.5f, 1.0f, 2.0f, 1.0f, 0.0f}).value();
  auto l1Biases = nnn::FloatMatrix::Create(3, 1, {0.0f, 0.5f, 1.0f}).value();

  nnn::DenseLayer* layer1 = GetLayerAs<nnn::DenseLayer>(&neuralNetwork, l1);
  layer1->Update(l1Weights, l1Biases);

  auto l2Weights = nnn::FloatMatrix::Identity(3);
  auto l2Biases = nnn::FloatMatrix::Create(3, 1, {0.0f, 1.0f, 0.0f}).value();

  nnn::SoftmaxDenseOutputLayer* layer2 = GetLayerAs<nnn::SoftmaxDenseOutputLayer>(&neuralNetwork, l2);
  layer2->Update(l2Weights, l2Biases);

  nnn::FloatMatrix input = nnn::FloatMatrix::Create(3, 1, {1.0f, 2.0f, 3.0f}).value();
  auto result = neuralNetwork.RunForwardPass(input);

  nnn::FloatMatrix expected = nnn::FloatMatrix::Create(3, 1, {1.0f, 2.0f, 1.0f}).value();
  nnn::CrossEntropyWithSoftmax crossEntropy;
  auto gradient = crossEntropy.Loss(result, expected);
  CHECK(gradient.GetRowCount() == 3);
  CHECK(gradient.GetColCount() == 1);
  CHECK_THAT(gradient(0, 0), Catch::Matchers::WithinAbs((0.211942f - 1.0f), 0.001));
  CHECK_THAT(gradient(1, 0), Catch::Matchers::WithinAbs((0.576117f - 2.0f), 0.001));
  CHECK_THAT(gradient(2, 0), Catch::Matchers::WithinAbs((0.211942f - 1.0f), 0.001));

  // TODO: does not make sense to check the results, weights initialization will change in the near future
  for (size_t i = 0; i < 10; i++) {
    neuralNetwork.RunBackwardPass(gradient);
    neuralNetwork.UpdateWeights();

    result = neuralNetwork.RunForwardPass(input);
    gradient = crossEntropy.Loss(result, expected);
  }
}

TEST_CASE("3 Layer NN - Solve XOR as a decision problem with ReLU and Softmax") {  //

  // intentionally testing overfitting
  auto neuralNetwork = nnn::NeuralNetwork(nnn::NeuralNetwork::HyperParameters(0.8f, 200));  // learning rate, epochs

  auto init = nnn::NormalGlorotWeightInitializer(42);

  size_t l1 = neuralNetwork.AddHiddenLayer(
      std::make_unique<nnn::DenseLayer>(2, 4, std::make_unique<nnn::LeakyReLU>(0.05f), init));
  size_t l2 = neuralNetwork.AddHiddenLayer(
      std::make_unique<nnn::DenseLayer>(4, 4, std::make_unique<nnn::LeakyReLU>(0.05f), init));

  size_t l3 = neuralNetwork.SetOutputLayer(std::make_unique<nnn::SoftmaxDenseOutputLayer>(4, 2, init));

  auto input = nnn::FloatMatrix::Create(2, 4, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f}).value();
  auto expected = nnn::FloatMatrix::Create(2, 4, {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f}).value();

  auto dataset = nnn::TrainingDataset(std::make_shared<nnn::FloatMatrix>(input),
      std::make_shared<nnn::FloatMatrix>(expected), {4, 0.0f});  // batch size, validation set %
  neuralNetwork.Train(dataset);

  auto result = neuralNetwork.RunForwardPass(input);

  // 99% confidence XOR(0,0) is 0
  CHECK_THAT(result(0, 0), Catch::Matchers::WithinAbs(1.0f, 0.01));
  CHECK_THAT(result(1, 0), Catch::Matchers::WithinAbs(0.0f, 0.01));

  // 99% confidence XOR(0,1) is 1
  CHECK_THAT(result(0, 1), Catch::Matchers::WithinAbs(0.0f, 0.01));
  CHECK_THAT(result(1, 1), Catch::Matchers::WithinAbs(1.0f, 0.01));

  // 99% confidence XOR(1,0) is 1
  CHECK_THAT(result(0, 2), Catch::Matchers::WithinAbs(0.0f, 0.01));
  CHECK_THAT(result(1, 2), Catch::Matchers::WithinAbs(1.0f, 0.01));

  // 99% confidence XOR(1,1) is 0
  CHECK_THAT(result(0, 3), Catch::Matchers::WithinAbs(1.0f, 0.01));
  CHECK_THAT(result(1, 3), Catch::Matchers::WithinAbs(0.0f, 0.01));
}

TEST_CASE("3 Layer NN - Solve XOR as a decision problem with ReLU and Softmax - From CSV") {  //

  // intentionally testing overfitting
  auto neuralNetwork = nnn::NeuralNetwork(nnn::NeuralNetwork::HyperParameters(0.8f, 200));  // learning rate, epochs

  auto init = nnn::NormalGlorotWeightInitializer(42);

  size_t l1 = neuralNetwork.AddHiddenLayer(
      std::make_unique<nnn::DenseLayer>(2, 4, std::make_unique<nnn::LeakyReLU>(0.05f), init));
  size_t l2 = neuralNetwork.AddHiddenLayer(
      std::make_unique<nnn::DenseLayer>(4, 4, std::make_unique<nnn::LeakyReLU>(0.05f), init));

  size_t l3 = neuralNetwork.SetOutputLayer(std::make_unique<nnn::SoftmaxDenseOutputLayer>(4, 2, init));

  auto reader = std::make_shared<nnn::CSVReader>();
  auto datasetResult = nnn::DataLoader::Load(
      {.trainingFeatures = "../../../../../src/lib/core/tests/xorTrainingFeatures.csv",
          .trainingLabels = "../../../../../src/lib/core/tests/xorTrainingLabels.csv",
          .testingFeatures = "../../../../../src/lib/core/tests/xorTestFeatures.csv",
          .testingLabels = "../../../../../src/lib/core/tests/xorTestLabels.csv"},
      reader, {.batchSize = 4, .validationSetFraction = 0.0f}, {.expectedClassNumber = 2, .shouldOneHotEncode = false});

  REQUIRE(datasetResult.has_value());

  neuralNetwork.Train(datasetResult.value().trainingDataset);

  auto result = neuralNetwork.RunForwardPass(*datasetResult.value().testingFeatures);

  // 99% confidence XOR(0,0) is 0
  CHECK_THAT(result(0, 0), Catch::Matchers::WithinAbs((*datasetResult.value().testingLabels)(0, 0), 0.01));
  CHECK_THAT(result(1, 0), Catch::Matchers::WithinAbs((*datasetResult.value().testingLabels)(1, 0), 0.01));

  // 99% confidence XOR(0,1) is 1
  CHECK_THAT(result(0, 1), Catch::Matchers::WithinAbs((*datasetResult.value().testingLabels)(0, 1), 0.01));
  CHECK_THAT(result(1, 1), Catch::Matchers::WithinAbs((*datasetResult.value().testingLabels)(1, 1), 0.01));

  // 99% confidence XOR(1,0) is 1
  CHECK_THAT(result(0, 2), Catch::Matchers::WithinAbs((*datasetResult.value().testingLabels)(0, 2), 0.01));
  CHECK_THAT(result(1, 2), Catch::Matchers::WithinAbs((*datasetResult.value().testingLabels)(1, 2), 0.01));

  // 99% confidence XOR(1,1) is 0
  CHECK_THAT(result(0, 3), Catch::Matchers::WithinAbs((*datasetResult.value().testingLabels)(0, 3), 0.01));
  CHECK_THAT(result(1, 3), Catch::Matchers::WithinAbs((*datasetResult.value().testingLabels)(1, 3), 0.01));
}

TEST_CASE("3 Layer NN - Recognize when a point is in a circle + Validation") {  //

  auto neuralNetwork = nnn::NeuralNetwork(nnn::NeuralNetwork::HyperParameters(0.3f, 80));  // learning rate, epochs

  auto initR = nnn::NormalGlorotWeightInitializer(42);
  auto initS = nnn::NormalHeWeightInitializer(42);

  size_t l1 =
      neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(2, 8, std::make_unique<nnn::LeakyReLU>(), initR));
  size_t l2 =
      neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(8, 8, std::make_unique<nnn::LeakyReLU>(), initR));
  size_t l3 = neuralNetwork.SetOutputLayer(std::make_unique<nnn::SoftmaxDenseOutputLayer>(8, 2, initS));

  auto input = nnn::FloatMatrix::Create(200, 2,
      {-0.1811f, 0.2951f, 0.3547f, -1.1966f, -1.0151f, 1.1957f, 1.4580f, 0.7601f, -0.9143f, 0.6674f, 1.3085f, 0.5881f,
          -0.4451f, -0.0399f, 0.2200f, 0.3281f, -0.0832f, -0.0056f, -0.9276f, 0.1636f, 0.8100f, -0.8525f, 0.7518f,
          0.9205f, 0.8949f, 0.4499f, -0.3798f, 0.0252f, 1.3244f, -0.3417f, 0.3868f, 1.1324f, -0.0792f, -0.3366f,
          1.1700f, -0.4860f, -0.0834f, 0.6178f, 0.1607f, -0.1796f, 1.3836f, 1.2161f, 0.3733f, 0.6843f, -0.2537f,
          0.4966f, 0.9281f, 0.9303f, -1.0710f, 0.7845f, -1.2167f, 0.5490f, -0.8774f, 0.0342f, 1.1761f, 0.3934f, 0.1409f,
          -0.0270f, -0.3712f, -1.2495f, -0.1106f, -0.8334f, 0.2102f, -1.2085f, 0.1480f, -0.6589f, 0.0297f, -0.0681f,
          0.0094f, 1.0695f, 0.5936f, -0.1064f, 0.4344f, 0.3596f, -0.2273f, 1.2191f, -0.5111f, -0.1405f, -0.4776f,
          1.2923f, 1.4715f, -0.2621f, 0.7138f, -0.5459f, 0.5005f, 0.6011f, -0.2731f, -0.9801f, -0.9126f, -1.2919f,
          0.0042f, 0.9527f, 1.1321f, 0.7223f, 0.4489f, 1.0477f, -0.7806f, -0.0131f, 0.4364f, -0.9687f, 0.6365f,
          -0.7883f, 0.2437f, -0.1836f, -1.1956f, 0.4905f, -1.0307f, -0.7493f, -0.5545f, -0.6497f, 0.5829f, -0.1865f,
          -1.1977f, -1.4453f, 0.9384f, 1.3417f, 0.2420f, -0.0805f, -0.4087f, -0.4481f, 0.3193f, -1.4724f, 0.7849f,
          0.3940f, -1.4205f, 0.2573f, -0.5399f, 1.1866f, -1.1188f, 0.0667f, 0.0598f, -0.8177f, 1.2446f, -0.3895f,
          -0.7694f, 0.3038f, 0.7371f, 0.2215f, -0.3324f, -1.4675f, 1.3646f, 0.7137f, -0.5421f, 1.3502f, 0.1126f,
          0.6362f, -0.1699f, -0.1092f, 0.9434f, -0.6544f, -0.8084f, -0.3276f, -1.4848f, -1.0176f, -0.0665f, -0.4829f,
          0.4235f, 0.2224f, -1.1664f, -0.0221f, 0.8187f, 0.0856f, 1.3909f, 1.0590f, 1.4190f, -0.3207f, -1.4085f,
          -1.3880f, -1.4239f, 1.3879f, 0.1209f, -0.3554f, 0.4512f, 0.3178f, 0.8856f, -0.6875f, 0.3450f, 1.4702f,
          -1.1906f, 1.2077f, 0.5797f, -0.6942f, 0.2200f, 0.5219f, -1.3620f, -1.3778f, 0.4261f, -1.2476f, -1.2883f,
          0.4273f, 0.5194f, 0.1226f, 0.6819f, -0.3645f, -0.1754f, 1.1631f, -0.6577f, -1.4271f, -0.5599f, -0.0302f,
          0.7526f, 0.7636f, -0.2019f, -0.7486f, -0.0969f, 0.1893f, -1.3273f, 0.1486f, 0.1758f, -0.9253f, 0.6801f,
          -0.5132f, -0.3733f, -1.2181f, -1.0797f, 0.0550f, -1.3311f, -1.1435f, 0.9678f, -0.4194f, 1.2161f, -1.2261f,
          -0.9915f, 0.1704f, 0.2721f, -0.0225f, -0.6494f, 0.3179f, -0.6539f, -0.9677f, -1.3450f, 0.0941f, 1.0664f,
          0.6110f, 0.2348f, -1.3922f, 1.1012f, 1.2397f, 0.1561f, -0.1000f, -0.7759f, -0.1774f, 0.0495f, 0.9916f,
          -0.1684f, 0.6353f, 0.1534f, -0.2583f, 1.0346f, -1.4302f, 1.0534f, -0.5492f, 0.6285f, 0.7360f, -0.2095f,
          -0.6891f, 0.0223f, -0.6402f, 0.4418f, 0.3820f, 0.0372f, 0.7268f, 0.5300f, 0.2237f, -0.8502f, -0.3073f,
          -1.4536f, 1.2850f, -1.1474f, 0.4476f, 1.3207f, 0.2264f, 0.1548f, -0.9724f, 1.3518f, 0.2203f, -1.4568f,
          -1.1518f, -0.0406f, 0.2232f, 0.3479f, 0.8262f, 0.4053f, -1.3641f, 0.3525f, -0.9173f, 0.1242f, 0.1000f,
          -0.2722f, -0.4288f, 0.0625f, 0.9510f, -0.1831f, -1.2646f, -0.3492f, 0.0106f, -0.6360f, -0.7042f, -0.5659f,
          0.7131f, 0.3687f, -1.2440f, -0.2154f, 1.4000f, -0.3609f, 0.4024f, -0.4310f, 0.8701f, -0.6295f, 0.5429f,
          0.6899f, -0.6699f, 0.2943f, 0.6783f, -0.3399f, 0.4728f, -1.2264f, -0.0167f, 0.6187f, -1.2560f, 0.1053f,
          0.7078f, 0.8747f, 0.8689f, 0.4469f, 0.3984f, 0.7052f, 0.9104f, 0.3675f, 0.7142f, -0.4473f, -1.1488f, -0.6639f,
          -0.6499f, -0.6964f, -0.4970f, -0.2216f, 0.6694f, -0.0775f, -1.2065f, 1.3214f, 1.3618f, 0.3443f, -0.9006f,
          -1.2190f, -0.3969f, -1.2455f, 1.4599f, -1.1455f, 0.5902f, -0.1161f, -0.1340f, 1.0079f, 0.5879f, -1.4659f,
          -0.0940f, 0.4761f, -1.0112f, 0.6059f, 0.8874f, -0.9160f, -0.2312f, 0.2849f, 0.4567f, -0.5539f, -0.0702f,
          0.1733f, 0.3896f, -0.5449f, -0.0855f, -0.7044f, -0.7680f, -0.9804f, -0.1984f, -0.2657f, -0.4338f, 0.6783f,
          1.4276f, 1.0752f, -0.2130f, 0.7652f, -0.5949f, -1.2477f, 0.6029f, 0.2783f, 0.8335f, 0.5501f, -0.5751f,
          -0.4282f, 0.2111f, -0.0110f, 0.9810f, -1.2817f, 0.9656f, 0.2139f, 0.8870f, 0.0990f, 0.2805f, -1.2864f,
          -0.5431f})
                   .value();

  input.MapInPlace([](float x) { return x / 1.5f; });

  auto expected = nnn::FloatMatrix::Create(200, 2,
      {0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
          1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f,
          1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
          0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
          1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
          1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
          1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f,
          0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
          1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
          0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
          1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
          1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
          0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
          0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
          1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
          0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
          1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
          1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f,
          1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
          1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
          1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f,
          0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
          1.0f, 1.0f, 0.0f})
                      .value();

  // use our column convention
  input.Transpose();
  expected.Transpose();

  auto dataset = nnn::TrainingDataset(std::make_shared<nnn::FloatMatrix>(input),
      std::make_shared<nnn::FloatMatrix>(expected), {45, 0.1f});  // batch size, validation set %
  auto statistics = neuralNetwork.Train(dataset);

  // 3 outside 2 inside 1 inside near boundary
  auto test = nnn::FloatMatrix::Create(2, 6, {1.0f, 0.9, -1.0f, 0.3f, -0.1f, 0.5f, -1.0f, 0.7, 0.2f, 0.2f, 0.4f, -0.5f})
                  .value();

  auto results = neuralNetwork.RunForwardPass(test);

  // outside
  CHECK_THAT(results(0, 0), Catch::Matchers::WithinAbs(1.0f, 0.01));
  CHECK_THAT(results(1, 0), Catch::Matchers::WithinAbs(0.0f, 0.01));

  CHECK_THAT(results(0, 1), Catch::Matchers::WithinAbs(1.0f, 0.01));
  CHECK_THAT(results(1, 1), Catch::Matchers::WithinAbs(0.0f, 0.01));

  CHECK_THAT(results(0, 2), Catch::Matchers::WithinAbs(1.0f, 0.01));
  CHECK_THAT(results(1, 2), Catch::Matchers::WithinAbs(0.0f, 0.01));

  // inside
  CHECK_THAT(results(0, 3), Catch::Matchers::WithinAbs(0.0f, 0.01));
  CHECK_THAT(results(1, 3), Catch::Matchers::WithinAbs(1.0f, 0.01));

  CHECK_THAT(results(0, 4), Catch::Matchers::WithinAbs(0.0f, 0.01));
  CHECK_THAT(results(1, 4), Catch::Matchers::WithinAbs(1.0f, 0.01));

  CHECK(results(1, 5) > results(0, 5));  // near the border
}
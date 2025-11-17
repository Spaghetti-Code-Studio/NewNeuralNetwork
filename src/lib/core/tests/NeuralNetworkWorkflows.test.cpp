#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <memory>

#include "CrossEntropyWithSoftmax.hpp"
#include "DenseLayer.hpp"
#include "FloatMatrix.hpp"
#include "ILayer.hpp"
#include "NeuralNetwork.hpp"
#include "ReLU.hpp"
#include "Softmax.hpp"
#include "SoftmaxDenseOutputLayer.hpp"

#include "TestableNeuralNetwork.hpp"

// ------------------------------------------------------------------------------------------------

/**
 * @brief Retrieves given layer from the neural network.
 *
 * @tparam LayerType type of the layer to retrieve
 * @param neuralNetwork neural network
 * @param index index of the layer in the network
 * @returns poiter to the layer or null pointer if the index is wrong or the type does not match
 */
template <typename LayerType>
static inline LayerType* GetLayerAs(nnn::NeuralNetwork* neuralNetwork, size_t index) {
  nnn::ILayer* baseLayer = neuralNetwork->GetLayer(index);
  return dynamic_cast<LayerType*>(baseLayer);
}

// ------------------------------------------------------------------------------------------------

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

// TEST_CASE("2 Layer NN - Train XOR with ReLU") {
//   auto neuralNetwork = nnn::NeuralNetwork();
//
//   size_t l1 = neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(4, 2, 2,
//   std::make_unique<nnn::ReLU>())); size_t l2 = neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(4, 2,
//   2, std::make_unique<nnn::ReLU>())); size_t l3 =
//       neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(4, 2, 1, std::make_unique<nnn::LeakyReLU>()));
//
//   auto weights1 = nnn::FloatMatrix::Random(2, 2);
//   auto biases1 = nnn::FloatMatrix::Random(1, 2);
//   biases1.Transpose();
//
//   nnn::ILayer* baseLayer1 = neuralNetwork.GetLayer(l1);
//   nnn::DenseLayer* denseLayer1 = dynamic_cast<nnn::DenseLayer*>(baseLayer1);
//   denseLayer1->Update(weights1, biases1);
//
//   auto weights2 = nnn::FloatMatrix::Random(2, 2);
//   auto biases2 = nnn::FloatMatrix::Random(1, 2);
//   biases2.Transpose();
//
//   nnn::ILayer* baseLayer2 = neuralNetwork.GetLayer(l2);
//   nnn::DenseLayer* denseLayer2 = dynamic_cast<nnn::DenseLayer*>(baseLayer2);
//   denseLayer2->Update(weights2, biases2);
//
//   auto weights3 = nnn::FloatMatrix::Random(1, 2);
//   auto biases3 = nnn::FloatMatrix::Random(1, 1);
//
//   nnn::ILayer* baseLayer3 = neuralNetwork.GetLayer(l3);
//   nnn::DenseLayer* denseLayer3 = dynamic_cast<nnn::DenseLayer*>(baseLayer3);
//   denseLayer3->Update(weights3, biases3);
//
//   // compute all combinations at once
//   auto input = nnn::FloatMatrix::Create(2, 4,
//       {
//           0.0f,
//           0.0f,
//           1.0f,
//           1.0f,
//           0.0f,
//           1.0f,
//           0.0f,
//           1.0f,
//       })
//                    .value();
//
//   auto expected = nnn::FloatMatrix::Create(1, 4, {0.0f, 1.0f, 1.0f, 0.0f}).value();
//
//   for (size_t i = 0; i < 250; i++) {
//     neuralNetwork.Train(input, expected, {0.075f});
//   }
//
//   auto result = neuralNetwork.RunForwardPass(input);
//
//   CHECK_THAT(result(0, 0), Catch::Matchers::WithinAbs(0.0f, 0.001));
//   CHECK_THAT(result(0, 1), Catch::Matchers::WithinAbs(1.0f, 0.001));
//   CHECK_THAT(result(0, 2), Catch::Matchers::WithinAbs(1.0f, 0.001));
//   CHECK_THAT(result(0, 3), Catch::Matchers::WithinAbs(0.0f, 0.001));
// }
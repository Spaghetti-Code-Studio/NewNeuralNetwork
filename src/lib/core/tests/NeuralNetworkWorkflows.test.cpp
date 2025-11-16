#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <memory>

#include "DenseLayer.hpp"
#include "FloatMatrix.hpp"
#include "ILayer.hpp"
#include "NeuralNetwork.hpp"
#include "ReLU.hpp"

#include "TestableNeuralNetwork.hpp"

TEST_CASE("1 Layer NN - Basic forward pass with ReLU") {  //

  auto neuralNetwork = TestableNeuralNetwork({true});
  size_t index = neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(3, 2, std::make_unique<nnn::ReLU>()));

  auto newBiases = nnn::FloatMatrix::Create(2, 1, {-5.0f, 0.5f}).value();
  auto newWeights = nnn::FloatMatrix::Create(2, 3, {1.0f, 2.0f, 0.0f, 0.5f, 0.5f, 1.0f}).value();

  nnn::ILayer* baseLayer = neuralNetwork.GetLayer(index);
  nnn::DenseLayer* layer = dynamic_cast<nnn::DenseLayer*>(baseLayer);

  // random state should not be tested
  layer->Update(newWeights, newBiases);

  nnn::FloatMatrix input = nnn::FloatMatrix::Create(3, 1, {1.0f, 2.0f, 3.0f}).value();
  auto result = neuralNetwork.RunForwardPass(input);

  CHECK_THAT(result(0, 0), Catch::Matchers::WithinAbs(0.0f, 0.001));
  CHECK_THAT(result(1, 0), Catch::Matchers::WithinAbs(5.0f, 0.001));
}

TEST_CASE("2 Layer NN - Forward pass XOR with ReLU") {
  auto neuralNetwork = TestableNeuralNetwork({true});

  size_t l1 = neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(2, 2, std::make_unique<nnn::ReLU>()));
  size_t l2 = neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(2, 1, std::make_unique<nnn::ReLU>()));

  auto weights1 = nnn::FloatMatrix::Create(2, 2, {1.0f, 1.0f, 1.0f, 1.0f}).value();
  auto biases1 = nnn::FloatMatrix::Create(2, 1, {-1.5f, -0.5f}).value();

  nnn::ILayer* baseLayer1 = neuralNetwork.GetLayer(l1);
  nnn::DenseLayer* denseLayer1 = dynamic_cast<nnn::DenseLayer*>(baseLayer1);
  denseLayer1->Update(weights1, biases1);

  auto weights2 = nnn::FloatMatrix::Create(1, 2, {-6.0f, 2.0f}).value();
  auto biases2 = nnn::FloatMatrix::Create(1, 1, {0.0f}).value();

  nnn::ILayer* baseLayer2 = neuralNetwork.GetLayer(l2);
  nnn::DenseLayer* denseLayer2 = dynamic_cast<nnn::DenseLayer*>(baseLayer2);
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

  nnn::ILayer* baseLayer1 = neuralNetwork.GetLayer(l1);
  nnn::DenseLayer* denseLayer1 = dynamic_cast<nnn::DenseLayer*>(baseLayer1);
  denseLayer1->Update(weights1, biases1);

  auto weights2 = nnn::FloatMatrix::Create(1, 2, {-6.0f, 2.0f}).value();
  auto biases2 = nnn::FloatMatrix::Zeroes(1, 1);

  nnn::ILayer* baseLayer2 = neuralNetwork.GetLayer(l2);
  nnn::DenseLayer* denseLayer2 = dynamic_cast<nnn::DenseLayer*>(baseLayer2);
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
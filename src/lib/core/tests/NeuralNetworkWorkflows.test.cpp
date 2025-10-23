#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <memory>

#include "DenseLayer.hpp"
#include "FloatMatrix.hpp"
#include "ILayer.hpp"
#include "NeuralNetwork.hpp"
#include "ReLU.hpp"

TEST_CASE("Neural Network Forward Pass Basic Dense Layer Forward Evaluation With ReLU") {  //

  auto relu = nnn::ReLU();
  nnn::FloatMatrix input = nnn::FloatMatrix::Create(3, 1, {1.0f, 2.0f, 3.0f}).value();

  auto neuralNetwork = nnn::NeuralNetwork();
  neuralNetwork.AddLayer(std::make_unique<nnn::DenseLayer>(3, 2, relu));

  auto result = neuralNetwork.RunForwardPass(input);

  CHECK_THAT(result(0, 0), Catch::Matchers::WithinAbs(4.820, 0.001));
  CHECK_THAT(result(1, 0), Catch::Matchers::WithinAbs(3.986, 0.001));

  // -----------------------------------------------------------------------------------------

  nnn::FloatMatrix newBiases = nnn::FloatMatrix::Create(2, 1, {-5.0f, 5.0f}).value();

  nnn::ILayer* baseLayer = neuralNetwork.GetLayer(0);
  nnn::DenseLayer* layer = dynamic_cast<nnn::DenseLayer*>(baseLayer);

  layer->Update(layer->GetWeights(), newBiases);

  result = neuralNetwork.RunForwardPass(input);

  CHECK_THAT(result(0, 0), Catch::Matchers::WithinAbs(0.0f, 0.001));
  CHECK_THAT(result(1, 0), Catch::Matchers::WithinAbs(8.986, 0.001));
}

// TEST_CASE("Basic DenseLayer Forward update") {
//   auto layer = nnn::DenseLayer(3, 2);
//   auto weights = nnn::FloatMatrix::Create(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
//   auto biases = nnn::FloatMatrix::Ones(2, 1);
//   layer.Update(weights.value(), biases);
//
//   auto inputResult = nnn::FloatMatrix::Create(3, 1, {1.0f, 2.0f, 3.0f});
//   auto& input = inputResult.value();
//   auto result = layer.Forward(input);
//
//   CHECK_THAT(result(0, 0), Catch::Matchers::WithinAbs(4.820, 0.001));
//   CHECK_THAT(result(1, 0), Catch::Matchers::WithinAbs(3.986, 0.001));
// }
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
#include "LeakyReLU.hpp"
#include "Softmax.hpp"
#include "SoftmaxDenseOutputLayer.hpp"
#include "NormalHeWeightInitializer.hpp"
#include "NormalGlorotWeightInitializer.hpp"

#include "TrainingDataset.hpp"

#include <iostream>

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
  auto neuralNetwork = nnn::NeuralNetwork(nnn::NeuralNetwork::HyperParameters(0.1f, 800));  // learning rate, epochs

  auto init = nnn::NormalGlorotWeightInitializer(42);

  size_t l1 =
      neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(2, 4, std::make_unique<nnn::LeakyReLU>(), init));
  size_t l2 =
      neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(4, 4, std::make_unique<nnn::LeakyReLU>(), init));

  size_t l3 = neuralNetwork.SetOutputLayer(std::make_unique<nnn::SoftmaxDenseOutputLayer>(4, 2, init));

  auto input = nnn::FloatMatrix::Create(2, 4, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f}).value();
  auto expected = nnn::FloatMatrix::Create(2, 4, {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f}).value();

  auto dataset = nnn::TrainingDataset(std::make_shared<nnn::FloatMatrix>(input),
      std::make_shared<nnn::FloatMatrix>(expected), {1, 0.0f});  // batch size, validation set %
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

TEST_CASE("Train a neural network to recognize when a circle is in a unit sphere + Validation") {  //

  // intentionally testing overfitting
  auto neuralNetwork = nnn::NeuralNetwork(nnn::NeuralNetwork::HyperParameters(0.04f, 2000));  // learning rate, epochs

  auto init = nnn::NormalGlorotWeightInitializer(42);

  size_t l1 =
      neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(3, 5, std::make_unique<nnn::LeakyReLU>(), init));
  size_t l2 =
      neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(5, 6, std::make_unique<nnn::LeakyReLU>(), init));
  size_t l3 =
      neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(6, 6, std::make_unique<nnn::LeakyReLU>(), init));
  size_t l4 =
      neuralNetwork.AddHiddenLayer(std::make_unique<nnn::DenseLayer>(6, 4, std::make_unique<nnn::LeakyReLU>(), init));
  size_t l5 = neuralNetwork.SetOutputLayer(std::make_unique<nnn::SoftmaxDenseOutputLayer>(4, 2, init));

  auto input = nnn::FloatMatrix::Create(3, 100,
      {0.197317f, -0.687963f, -0.688011f, -0.633191f, -0.391516f, 0.049513f, -0.136110f, -0.417542f, 0.223706f,
          -0.721012f, -0.415711f, -0.267276f, -0.087860f, 0.570352f, -0.600652f, 0.028469f, 0.184829f, -0.907099f,
          -0.390772f, -0.804656f, 0.368466f, -0.119695f, -0.755924f, -0.009646f, 0.325045f, -0.376578f, 0.040136f,
          -0.222645f, -0.457302f, 0.657475f, -0.286493f, -0.438131f, 0.085392f, 0.246596f, -0.338204f, -0.872883f,
          -0.378035f, -0.349633f, 0.459212f, 0.275115f, 0.774425f, -0.055570f, 0.122554f, 0.541934f, -0.012409f,
          0.045466f, -0.144918f, -0.949162f, -0.371288f, 0.017141f, 0.815133f, -0.501416f, -0.179234f, 0.511102f,
          0.266808f, 0.742921f, 0.607344f, -0.779896f, -0.544130f, -0.145784f, 0.021495f, -0.165178f, -0.555784f,
          -0.353594f, 0.037581f, 0.406038f, -0.496435f, -0.005503f, -0.398243f, 0.344271f, 0.523239f, -0.524725f,
          0.456433f, -0.264434f, 0.264612f, 0.267059f, 0.071549f, -0.819420f, 0.670605f, -0.358440f, -0.626963f,
          0.290346f, -0.651267f, 0.381875f, 0.754679f, -0.484117f, 0.319968f, 0.634444f, 0.110402f, 0.059301f,
          0.800836f, 0.266203f, -0.321940f, -0.301581f, 0.451911f, 0.794221f, 0.774173f, 0.559751f, 0.284063f,
          0.097468f, 0.383790f, 0.303923f, -0.551461f, 0.424358f, -0.525502f, -0.349201f, 0.492983f, 0.299266f,
          0.698447f, 0.315226f, 0.136617f, -0.812650f, -0.264568f, -0.469595f, 0.005274f, 0.153808f, -0.014965f,
          -0.609514f, 0.444904f, -0.438455f, 0.706019f, -0.411102f, -0.229805f, 0.113603f, 0.872310f, 0.392060f,
          0.140122f, -0.805647f, 0.230014f, 0.754746f, 0.481537f, 0.394031f, 0.404968f, -0.281018f, -0.412816f,
          0.826481f, 0.022685f, 0.003033f, 0.596590f, 0.299928f, 0.403934f, -0.248834f, -0.812036f, 0.156560f,
          -0.928115f, -0.068804f, 0.085289f, -0.745879f, 0.044487f, 0.539987f, -0.210278f, -0.204777f, 1.068324f,
          -0.997812f, -0.199159f, -0.130602f, -0.014958f, -0.098820f, -1.314922f, -0.334022f, 0.676903f, -1.006565f,
          -0.256912f, 0.103644f, 1.101101f, 0.485348f, -0.116043f, -0.881182f, -0.586996f, 0.377959f, 1.153106f,
          -0.001626f, 1.065349f, -0.166083f, -0.559185f, -0.883554f, 0.865391f, -0.110612f, -1.451737f, -0.251060f,
          -0.802266f, 0.443678f, 0.929941f, 0.002624f, -0.053184f, 1.176782f, 0.129410f, 0.038483f, 1.049387f,
          -0.408523f, -1.363134f, 0.115712f, -1.043826f, 0.055039f, 0.087317f, -0.943986f, 0.416611f, 0.340629f,
          -0.122608f, -0.139236f, 1.294700f, -0.834963f, -0.844035f, -0.011698f, -0.379684f, -0.587571f, 1.245184f,
          -0.053884f, -0.067189f, 1.031696f, 1.169131f, -0.460937f, -0.303691f, -0.735596f, -0.927551f, 0.156159f,
          1.112465f, -0.428655f, 0.445779f, 0.707679f, -0.478663f, 1.209229f, 0.047722f, 0.035030f, 1.032986f,
          -0.094903f, -0.211985f, 1.021141f, 0.047551f, -0.070087f, 1.156390f, -0.101493f, 0.500295f, 1.311378f,
          -0.349036f, -0.366773f, -1.249700f, 0.349281f, -1.000011f, 0.864947f, 0.002398f, -0.620870f, -0.894329f,
          -1.173775f, 0.718097f, 0.585124f, -0.161811f, 0.252150f, -1.355491f, -0.909183f, 0.434869f, -1.013363f,
          0.330955f, 0.250522f, -1.313234f, 0.488810f, -0.938009f, 0.671026f, -0.037815f, 0.031604f, 1.446923f,
          1.028849f, 0.664655f, 0.781034f, 1.367351f, -0.438519f, -0.337296f, -0.993366f, 0.333525f, 0.796007f,
          -0.382282f, -0.722207f, -0.829426f, 0.097188f, -0.382330f, 1.338883f, 1.152439f, 0.435957f, -0.193283f,
          0.829216f, -0.706380f, 0.551087f, 0.449026f, 0.564038f, -0.775068f, 0.275174f, 0.202906f, 1.263675f,
          0.643319f, 0.316478f, -1.144461f, 0.310838f, 0.174331f, 1.305345f, -0.965481f, 0.974368f, 0.590303f,
          0.058467f, -0.020121f, -1.405040f})
                   .value();

  auto expected = nnn::FloatMatrix::Create(2, 100,
      {0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f,
          0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f,
          0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f,
          0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f,
          0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f,
          0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f,
          0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f,
          0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f,
          0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f,
          0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f,
          1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f,
          1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f,
          1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f,
          1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f,
          1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f,
          1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f,
          1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f,
          1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f,
          1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f,
          1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f, 1.000000f, 0.000000f})
                      .value();

  auto dataset = nnn::TrainingDataset(std::make_shared<nnn::FloatMatrix>(input),
      std::make_shared<nnn::FloatMatrix>(expected), {1, 0.15f});  // batch size, validation set %
  auto statistics = neuralNetwork.Train(dataset);

  // validation loss should be decreasing
  for (size_t i = 0; i < statistics.losses.size(); i++) {
    if (i % 200 == 0) std::cout << '+e' << i << ": " << statistics.losses[i] << '\n';
  }
  std::cout << std::endl;
}
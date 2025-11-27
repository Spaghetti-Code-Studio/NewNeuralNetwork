#pragma once

#include <filesystem>
#include <string>

#include <result.hpp>

namespace nnn {

  class Config {
   public:
    struct LayerAbstraction {
      size_t inputNumber;
      size_t outputNumber;
    };

    int randomSeed = 42;
    float learningRate = 0.01f;
    size_t epochs = 10;
    size_t batchSize = 256;
    float validationSetFraction = 0.2;
    size_t expectedClassNumber = 10;
    std::vector<LayerAbstraction> layers = {};

    Config() = default;
    cpp::result<void, std::string> LoadFromJSON(std::filesystem::path configFilePath);
    std::string ToString() const;
  };
}  // namespace nnn

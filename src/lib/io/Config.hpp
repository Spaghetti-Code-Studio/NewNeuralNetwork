#pragma once

#include <filesystem>
#include <string>
#include <thread>

#include <result.hpp>

namespace nnn {

  class Config {
   public:
    int randomSeed = 42;
    int hardThreadsLimit = std::thread::hardware_concurrency();
    float learningRate = 0.01f;
    float learningRateDecay = 1.0f;
    float weightDecay = 1.0f;
    float momentum = 0.0f;
    size_t epochs = 10;
    size_t batchSize = 256;
    float validationSetFraction = 0.2;
    size_t expectedClassNumber = 10;
    std::vector<size_t> layers = {};

    Config() = default;
    cpp::result<void, std::string> LoadFromJSON(std::filesystem::path configFilePath);
    std::string ToString() const;
  };
}  // namespace nnn

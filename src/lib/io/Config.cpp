#include "Config.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>

#include <json.hpp>

cpp::result<void, std::string> nnn::Config::LoadFromJSON(std::filesystem::path configFilePath) {  //

  if (!std::filesystem::exists(configFilePath)) {
    try {
      std::filesystem::path absolute_filepath = std::filesystem::absolute(configFilePath);
      return cpp::fail("File <" + absolute_filepath.string() + "> was not found!");
    } catch (const std::filesystem::filesystem_error& e) {
      return cpp::fail("Error resolving path: <" + configFilePath.string() + ">. Details: " + e.what());
    }
  }

  if (!std::filesystem::is_regular_file(configFilePath)) {
    return cpp::fail("Path <" + configFilePath.string() + "> exists but is not a regular file!");
  }

  std::ifstream file(configFilePath.string());
  nlohmann::json config;
  file >> config;

  try {
    randomSeed = config.value("randomSeed", 42);
  } catch (const nlohmann::json::exception& e) {
    return cpp::fail("Failed to parse 'randomSeed': " + std::string(e.what()));
  }

  try {
    learningRate = config.value("learningRate", 0.01f);
  } catch (const nlohmann::json::exception& e) {
    return cpp::fail("Failed to parse 'learningRate': " + std::string(e.what()));
  }

  try {
    learningRateDecay = config.value("learningRateDecay", 0.01f);
  } catch (const nlohmann::json::exception& e) {
    return cpp::fail("Failed to parse 'learningRateDecay': " + std::string(e.what()));
  }

  try {
    epochs = config.value("epochs", 10);
  } catch (const nlohmann::json::exception& e) {
    return cpp::fail("Failed to parse 'epochs': " + std::string(e.what()));
  }

  try {
    batchSize = config.value("batchSize", 256);
  } catch (const nlohmann::json::exception& e) {
    return cpp::fail("Failed to parse 'batchSize': " + std::string(e.what()));
  }

  try {
    validationSetFraction = config.value("validationSetFraction", 0.2);
  } catch (const nlohmann::json::exception& e) {
    return cpp::fail("Failed to parse 'validationSetFraction': " + std::string(e.what()));
  }

  try {
    const auto& layer_sizes_array = config.value("layers", nlohmann::json::array());

    if (layer_sizes_array.size() < 2) {
      return cpp::fail("'layers' array must contain at least 2 values (input and output).");
    }

    for (const auto& size_json : layer_sizes_array) {
      if (!size_json.is_number_unsigned()) {
        return cpp::fail("All values in 'layers' must be positive integers.");
      }
      layers.push_back(size_json.get<size_t>());
    }

  } catch (const nlohmann::json::exception& e) {
    return cpp::fail("Failed to parse 'layers' array: " + std::string(e.what()));
  }

  if (layers.size() > 0) {
    expectedClassNumber = layers.back();
  }

  return {};
}

std::string nnn::Config::ToString() const {  //

  std::ostringstream oss;

  oss << std::fixed << std::setprecision(4);

  oss << "--- NewNeuralNetwork Config ---\n";
  oss << "General settings:\n";
  oss << "  Random seed:            " << randomSeed << "\n";
  oss << "  Learning rate:          " << learningRate << "\n";
  oss << "  Learning rate decay:    " << learningRateDecay << "\n";
  oss << "  Epochs:                 " << epochs << "\n";
  oss << "  Batch size:             " << batchSize << "\n";
  oss << "  Validation fraction:    " << validationSetFraction << "\n";
  oss << "  Expected classes:       " << expectedClassNumber << "\n";

  oss << "\nLayers (total " << layers.size() - 1 << " layers):\n";

  if (layers.empty()) {
    oss << "  (No layers found)\n";
  } else {
    for (size_t i = 0; i < layers.size() - 1; ++i) {
      oss << "  Layer " << i + 1 << ": " << layers[i] << " -> " << layers[i + 1] << "\n";
    }
  }

  return oss.str();
}

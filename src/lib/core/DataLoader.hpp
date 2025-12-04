#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <result.hpp>

#include "FloatMatrix.hpp"
#include "IReader.hpp"
#include "TrainingDataset.hpp"

namespace nnn::DataLoader {

  struct Dataset {
    TrainingDataset trainingDataset;
    std::shared_ptr<FloatMatrix> testingFeatures;
    std::shared_ptr<FloatMatrix> testingLabels;
  };

  struct Filepaths {
    std::filesystem::path trainingFeatures;
    std::filesystem::path trainingLabels;
    std::filesystem::path testingFeatures;
    std::filesystem::path testingLabels;
  };

  struct TrainingParameters {
    size_t batchSize = 64;
    float validationSetFraction = 0.25f;
  };

  struct LoadingParameters {
    size_t expectedClassNumber = 2;
    bool shouldOneHotEncode = false;
    float normalizationFactor = 1.0f;
  };

  cpp::result<Dataset, std::string> Load(const Filepaths& filepaths,
      std::shared_ptr<IReader> reader,
      TrainingParameters trainingParams,
      LoadingParameters loadingParams);
}  // namespace nnn::DataLoader

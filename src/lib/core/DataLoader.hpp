#pragma once

#include <filesystem>
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
    size_t expectedClassNumber;
    size_t batchSize = 64;
    float validationSetFraction = 0.25f;
  };

  // TODO: optimally, this function should take clean-up logic and other functionality as parameter
  cpp::result<Dataset, std::string> Load(
      const Filepaths& filepaths, std::shared_ptr<IReader> reader, TrainingParameters options);
}  // namespace nnn::DataLoader

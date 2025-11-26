#include "DataLoader.hpp"

cpp::result<nnn::DataLoader::Dataset, std::string> nnn::DataLoader::Load(
    const Filepaths& filepaths, std::shared_ptr<IReader> reader, TrainingParameters options) {  //

  auto trainingFeaturesReadResult = reader->Read(filepaths.trainingFeatures);
  if (trainingFeaturesReadResult.has_error()) {
    return cpp::fail(trainingFeaturesReadResult.error());
  }

  auto trainingLabelsReadResult = reader->Read(filepaths.trainingLabels);
  if (trainingLabelsReadResult.has_error()) {
    return cpp::fail(trainingLabelsReadResult.error());
  }

  auto testingFeaturesReadResult = reader->Read(filepaths.testingFeatures);
  if (testingFeaturesReadResult.has_error()) {
    return cpp::fail(testingFeaturesReadResult.error());
  }

  auto testingLabelsReadResult = reader->Read(filepaths.testingLabels);
  if (testingLabelsReadResult.has_error()) {
    return cpp::fail(testingLabelsReadResult.error());
  }

  TrainingDataset trainingDataset(trainingFeaturesReadResult.value(), trainingLabelsReadResult.value(),
      {.batchSize = options.batchSize, .validationSetFraction = options.validationSetFraction});

  nnn::DataLoader::Dataset finalDataset = {.trainingDataset = trainingDataset,
      .testingFeatures = testingFeaturesReadResult.value(),
      .testingLabels = testingLabelsReadResult.value()};

  // TODO: normalization, one-hot encoding

  return finalDataset;
}

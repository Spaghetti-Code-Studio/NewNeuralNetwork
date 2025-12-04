#include "DataLoader.hpp"

cpp::result<nnn::DataLoader::Dataset, std::string> nnn::DataLoader::Load(const Filepaths& filepaths,
    std::shared_ptr<IReader> reader,
    TrainingParameters trainingParams,
    LoadingParameters loadingParams) {  //

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

  const float normFact = loadingParams.normalizationFactor;
  if (normFact != 1.0f && normFact != 0.0f) {
    trainingFeaturesReadResult.value()->MapInPlace([normFact](float x) { return x / normFact; });
    testingFeaturesReadResult.value()->MapInPlace([normFact](float x) { return x / normFact; });
  }

  // TODO: this could probably be done more efficiently by not loading the whole labels file, just reading it and
  // creating one-hot encoded data right away
  if (loadingParams.shouldOneHotEncode) {  //

    auto rowsTrain = trainingLabelsReadResult.value()->GetRowCount();
    auto rowsTest = testingLabelsReadResult.value()->GetRowCount();

    auto newTrainLabels = std::make_shared<nnn::FloatMatrix>(rowsTrain, loadingParams.expectedClassNumber);
    auto newTestLabels = std::make_shared<nnn::FloatMatrix>(rowsTest, loadingParams.expectedClassNumber);

    auto& oldTrainingLabelsPtr = *trainingLabelsReadResult.value();
    auto& oldTestingLabelsPtr = *testingLabelsReadResult.value();
    auto& newTrainLabelsPtr = *newTrainLabels;
    auto& newTestLabelsPtr = *newTestLabels;

    for (size_t row = 0; row < rowsTrain; row++) {
      newTrainLabelsPtr(row, static_cast<size_t>(oldTrainingLabelsPtr(row, 0))) = 1.0f;
    }

    for (size_t row = 0; row < rowsTest; row++) {
      newTestLabelsPtr(row, static_cast<size_t>(oldTestingLabelsPtr(row, 0))) = 1.0f;
    }

    trainingLabelsReadResult = newTrainLabels;
    testingLabelsReadResult = newTestLabels;
  }

  // adjust for column convention
  trainingFeaturesReadResult.value()->Transpose();
  testingFeaturesReadResult.value()->Transpose();
  trainingLabelsReadResult.value()->Transpose();
  testingLabelsReadResult.value()->Transpose();

  TrainingDataset trainingDataset(trainingFeaturesReadResult.value(), trainingLabelsReadResult.value(),
      {.batchSize = trainingParams.batchSize, .validationSetFraction = trainingParams.validationSetFraction});

  nnn::DataLoader::Dataset finalDataset = {.trainingDataset = trainingDataset,
      .testingFeatures = testingFeaturesReadResult.value(),
      .testingLabels = testingLabelsReadResult.value()};

  return finalDataset;
}

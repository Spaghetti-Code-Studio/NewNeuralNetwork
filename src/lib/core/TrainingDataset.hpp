#pragma once

#include <memory>
#include <random>

#include "FloatMatrix.hpp"

namespace nnn {

  class TrainingBatchGenerator;

  /**
   * @brief Groups data necessary for neural network training: feature and labels vectors. Divides dataset to both
   * training and validation subsets. Supports each batching of training vectors.
   */
  class TrainingDataset {
   public:
    struct TrainingDatasetParameters {
      size_t batchSize = 64;
      float validationSetFraction = 0.25f;
    };

    struct TrainingBatch {
      FloatMatrix features;
      FloatMatrix labels;
    };

    TrainingDataset(
        std::shared_ptr<FloatMatrix> features, std::shared_ptr<FloatMatrix> labels, TrainingDatasetParameters params);

    std::shared_ptr<const FloatMatrix> GetFeatures() const;
    std::shared_ptr<const FloatMatrix> GetLabels() const;

    FloatMatrix GetTrainingFeatures() const;
    FloatMatrix GetTrainingLabels() const;

    FloatMatrix GetValidationFeatures() const;
    FloatMatrix GetValidationLabels() const;
    bool HasValidationDataset() const;

    friend class TrainingBatchGenerator;

   private:
    std::shared_ptr<const FloatMatrix> m_features;
    std::shared_ptr<const FloatMatrix> m_labels;
    TrainingDatasetParameters m_params;
    size_t m_validationDatasetSize = 0;
    size_t m_trainingDatasetSize = 0;
    size_t m_trainingBatchCount = 0;
    size_t m_trainingBatchIndex = 0;
  };

  // TODO: this could be better done using ITrainingBatchGenerator or something.
  class TrainingBatchGenerator {
   public:
    struct TrainingBatchGeneratorParameters {
      bool isDataShufflingEnabled = false;
      int seed = 42;
    };

    TrainingBatchGenerator(TrainingDataset& dataset, TrainingBatchGeneratorParameters params);

    TrainingDataset::TrainingBatch GetNextBatch();
    bool HasNextBatch() const;
    void Reset();
    const std::vector<size_t>& GetIndices() const;

   private:
    TrainingDataset& m_dataset;
    TrainingBatchGeneratorParameters m_params;
    std::vector<size_t> m_indices;
    std::mt19937 m_generator;
  };
}  // namespace nnn

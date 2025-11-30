#pragma once

#include <memory>

#include "FloatMatrix.hpp"

namespace nnn {
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

    TrainingBatch GetNextBatch();
    bool HasNextBatch() const;
    void Reset();

   private:
    std::shared_ptr<const FloatMatrix> m_features;
    std::shared_ptr<const FloatMatrix> m_labels;
    TrainingDatasetParameters m_params;
    size_t m_validationDatasetSize = 0;
    size_t m_trainingDatasetSize = 0;
    size_t m_trainingBatchCount = 0;
    size_t m_trainingBatchIndex = 0;
  };
}  // namespace nnn

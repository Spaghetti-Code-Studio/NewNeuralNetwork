#pragma once

#include <memory>

#include "FloatMatrix.hpp"

namespace nnn {
  class TrainingDataset {
   public:
    struct TrainingDatasetParameters {
      size_t batchSize = 64;
      float validationSetFraction = 0.25f;

      TrainingDatasetParameters(size_t batchSize, float validationSetFraction)
          : batchSize(batchSize), validationSetFraction(validationSetFraction) {}
    };

    struct TrainingBatch {
      FloatMatrix features;
      FloatMatrix labels;
    };

    TrainingDataset(std::shared_ptr<const FloatMatrix> data,
        std::shared_ptr<const FloatMatrix> datalabels,
        TrainingDatasetParameters params);

    TrainingBatch GetNextBatch();
    bool HasNextBatch() const;
    bool HasValidationDataset() const { return m_params.validationSetFraction != 0.0f; }

    void Reset();
    const FloatMatrix& GetValidationFeatures() const;
    const FloatMatrix& GetValidationLabels() const;

    const FloatMatrix& GetTrainingFeatures() const;
    const FloatMatrix& GetTrainingLabels() const;

   private:
    std::shared_ptr<const FloatMatrix> m_data;
    std::shared_ptr<const FloatMatrix> m_labels;
    TrainingDatasetParameters m_params;
    size_t m_validationDatasetSize = 0;
    size_t m_trainingDatasetSize = 0;
    size_t m_trainingBatchCount = 0;
    size_t m_trainingBatchIndex = 0;
  };
}  // namespace nnn

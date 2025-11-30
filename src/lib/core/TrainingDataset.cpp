#include "TrainingDataset.hpp"

namespace nnn {

  TrainingDataset::TrainingDataset(
      std::shared_ptr<FloatMatrix> features, std::shared_ptr<FloatMatrix> labels, TrainingDatasetParameters params)
      : m_features(features), m_labels(labels), m_params(params) {  //

    int datasetSize = m_features->GetColCount();
    int batchSize = m_params.batchSize;

    int validationBatchCount = (m_params.validationSetFraction * datasetSize) / batchSize;
    int totalBatchCount = datasetSize / batchSize;
    int residueSize = datasetSize - (totalBatchCount * batchSize);

    // the residue should be added to the validation set to make batching easier
    m_validationDatasetSize = validationBatchCount * batchSize + residueSize;
    m_trainingDatasetSize = datasetSize - m_validationDatasetSize;
    m_trainingBatchCount = m_trainingDatasetSize / batchSize;
  }

  TrainingDataset::TrainingBatch TrainingDataset::GetNextBatch() {  //

    int currentIndex = m_trainingBatchIndex++ % m_trainingBatchCount;
    return {m_features->GetColumns(currentIndex * m_params.batchSize, (currentIndex + 1) * m_params.batchSize - 1),
        m_labels->GetColumns(currentIndex * m_params.batchSize, (currentIndex + 1) * m_params.batchSize - 1)};
  }

  bool TrainingDataset::HasNextBatch() const { return m_trainingBatchIndex < m_trainingBatchCount; }

  FloatMatrix TrainingDataset::GetValidationFeatures() const {
    return m_features->GetColumns(m_trainingDatasetSize, m_features->GetColCount() - 1);
  }
  FloatMatrix TrainingDataset::GetValidationLabels() const {
    return m_labels->GetColumns(m_trainingDatasetSize, m_labels->GetColCount() - 1);
  }

  bool TrainingDataset::HasValidationDataset() const { return m_params.validationSetFraction != 0.0f; }

  FloatMatrix TrainingDataset::GetTrainingFeatures() const {
    return m_features->GetColumns(0, m_trainingDatasetSize - 1);
  }

  FloatMatrix TrainingDataset::GetTrainingLabels() const { return m_labels->GetColumns(0, m_trainingDatasetSize - 1); }

  std::shared_ptr<const FloatMatrix> TrainingDataset::GetFeatures() const { return m_features; }

  std::shared_ptr<const FloatMatrix> TrainingDataset::GetLabels() const { return m_labels; }

  void TrainingDataset::Reset() { m_trainingBatchIndex = 0; }
}  // namespace nnn

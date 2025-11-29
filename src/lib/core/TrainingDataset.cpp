#include "TrainingDataset.hpp"

namespace nnn {

  TrainingDataset::TrainingDataset(
      std::shared_ptr<FloatMatrix> data, std::shared_ptr<FloatMatrix> labels, TrainingDatasetParameters params)
      : m_data(data), m_labels(labels), m_params(params) {  //

    int datasetSize = m_data->GetColCount();
    int batchSize = m_params.batchSize;

    m_trainingBatchCount = datasetSize / batchSize;
    int residueSize = datasetSize - (m_trainingBatchCount * batchSize);
    int sizeWithoutResidue = datasetSize - residueSize;
    int validationBatchCount = (m_params.validationSetFraction * datasetSize) / batchSize;

    m_validationDatasetSize = validationBatchCount * batchSize + residueSize;
    m_trainingDatasetSize = datasetSize - m_validationDatasetSize;
  }

  TrainingDataset::TrainingBatch TrainingDataset::GetNextBatch() {
    int currentIndex = m_trainingBatchIndex++ % m_trainingBatchCount;
    return {m_data->GetColumns(currentIndex * m_params.batchSize, (currentIndex + 1) * m_params.batchSize - 1),
        m_labels->GetColumns(currentIndex * m_params.batchSize, (currentIndex + 1) * m_params.batchSize - 1)};
  }

  bool TrainingDataset::HasNextBatch() const { return m_trainingBatchIndex < m_trainingBatchCount; }

  FloatMatrix TrainingDataset::GetValidationFeatures() const {
    FloatMatrix validationFeatures = m_data->GetColumns(m_trainingDatasetSize, m_data->GetColCount() - 1);
    return validationFeatures;
  }
  FloatMatrix TrainingDataset::GetValidationLabels() const {
    FloatMatrix validationLabels = m_labels->GetColumns(m_trainingDatasetSize, m_data->GetColCount() - 1);
    return validationLabels;
  }

  FloatMatrix TrainingDataset::GetTrainingFeatures() const {
    FloatMatrix trainingFeatures = m_data->GetColumns(0, m_trainingDatasetSize - 1);
    return trainingFeatures;
  }

  FloatMatrix TrainingDataset::GetTrainingLabels() const {
    FloatMatrix trainingLabels = m_labels->GetColumns(0, m_trainingDatasetSize - 1);
    return trainingLabels;
  }

  void TrainingDataset::Reset() { m_trainingBatchIndex = 0; }
}  // namespace nnn

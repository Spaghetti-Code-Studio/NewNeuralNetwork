#include "TrainingDataset.hpp"

#include <algorithm>
#include <numeric>

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

  // -----------------------------------------------------------------------------------------------------------

  TrainingBatchGenerator::TrainingBatchGenerator(TrainingDataset& dataset, TrainingBatchGeneratorParameters params)
      : m_dataset(dataset), m_params(params), m_indices({}), m_generator(params.seed) {  //

    if (m_params.isDataShufflingEnabled) {
      m_indices.resize(m_dataset.m_trainingDatasetSize);
      std::iota(m_indices.begin(), m_indices.end(), 0);
      std::shuffle(m_indices.begin(), m_indices.end(), m_generator);
    }
  }

  TrainingBatchGenerator::TrainingBatch TrainingBatchGenerator::GetNextBatch() {  //

    int currentIndex = m_dataset.m_trainingBatchIndex++ % m_dataset.m_trainingBatchCount;
    if (!m_params.isDataShufflingEnabled) {
      return {m_dataset.m_features->GetColumns(
                  currentIndex * m_dataset.m_params.batchSize, (currentIndex + 1) * m_dataset.m_params.batchSize - 1),
          m_dataset.m_labels->GetColumns(
              currentIndex * m_dataset.m_params.batchSize, (currentIndex + 1) * m_dataset.m_params.batchSize - 1)};
    } else {
      std::vector<size_t> subvector(m_indices.begin() + currentIndex * m_dataset.m_params.batchSize,
          m_indices.begin() + (currentIndex + 1) * m_dataset.m_params.batchSize);
      return {m_dataset.m_features->GetColumns(subvector), m_dataset.m_labels->GetColumns(subvector)};
    }
  }

  bool TrainingBatchGenerator::HasNextBatch() const {
    return m_dataset.m_trainingBatchIndex < m_dataset.m_trainingBatchCount;
  }

  void TrainingBatchGenerator::Reset() {
    m_dataset.m_trainingBatchIndex = 0;
    if (m_params.isDataShufflingEnabled) {
      std::shuffle(m_indices.begin(), m_indices.end(), m_generator);
    }
  }
  const std::vector<size_t>& TrainingBatchGenerator::GetIndices() const { return m_indices; }
}  // namespace nnn

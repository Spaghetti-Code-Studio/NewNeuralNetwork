#include "FloatMatrix.hpp"
#include "RowMajorFloatMatrixIterator.hpp"

namespace nnn {
  RowMajorFloatMatrixIterator::RowMajorFloatMatrixIterator(FloatMatrix* mat) : m_matrix(mat), m_row(0), m_col(0) {}

  void RowMajorFloatMatrixIterator::Restart() {
    m_row = 0;
    m_col = 0;
  }

  float& RowMajorFloatMatrixIterator::Get() { return (*m_matrix)(m_row, m_col); }

  const float& RowMajorFloatMatrixIterator::Get() const { return (*m_matrix)(m_row, m_col); }

  void RowMajorFloatMatrixIterator::Next() {
    ++m_col;
    if (m_col >= m_matrix->GetColCount()) {
      m_col = 0;
      ++m_row;
    }
  }

  bool RowMajorFloatMatrixIterator::HasNext() const { return m_row < m_matrix->GetRowCount(); }

}  // namespace nnn

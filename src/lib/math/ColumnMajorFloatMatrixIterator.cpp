#include "ColumnMajorFloatMatrixIterator.hpp"
#include "FloatMatrix.hpp"

namespace nnn {
  ColumnMajorFloatMatrixIterator::ColumnMajorFloatMatrixIterator(const FloatMatrix* mat, size_t row, size_t col)
      : m_matrix(mat), m_row(row), m_col(col) {}

  ColumnMajorFloatMatrixIterator::ColumnMajorFloatMatrixIterator(const FloatMatrix* mat)
      : m_matrix(mat), m_row(0), m_col(0) {}

  // TODO: keep original starting row and col and set them here
  void ColumnMajorFloatMatrixIterator::Restart() {
    m_row = 0;
    m_col = 0;
  }

  const float& ColumnMajorFloatMatrixIterator::Get() const { return (*m_matrix)(m_row, m_col); }

  void ColumnMajorFloatMatrixIterator::Next() {
    ++m_row;
    if (m_row >= m_matrix->GetRowCount()) {
      m_row = 0;
      ++m_col;
    }
  }

  bool ColumnMajorFloatMatrixIterator::HasNext() const { return m_col < m_matrix->GetColCount(); }
}  // namespace nnn

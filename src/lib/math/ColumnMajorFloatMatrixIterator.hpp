#pragma once
#include <cstddef>

namespace nnn {

  class FloatMatrix;

  class ColumnMajorFloatMatrixIterator {
   private:
    const FloatMatrix* m_matrix;
    size_t m_row;
    size_t m_col;

   public:
    ColumnMajorFloatMatrixIterator(const FloatMatrix* mat);

    void Restart();

    const float& Get() const;

    void Next();
    bool HasNext() const;
  };
}  // namespace nnn

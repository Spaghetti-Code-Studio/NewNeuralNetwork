#pragma once
#include <cstddef>

namespace nnn {

  class FloatMatrix;

  class RowMajorFloatMatrixIterator {
   private:
    FloatMatrix* m_matrix;
    size_t m_row;
    size_t m_col;

   public:
    RowMajorFloatMatrixIterator(FloatMatrix* mat);

    void Restart();

    float& Get();
    const float& Get() const;

    void Next();
    bool HasNext() const;
  };
}  // namespace nnn

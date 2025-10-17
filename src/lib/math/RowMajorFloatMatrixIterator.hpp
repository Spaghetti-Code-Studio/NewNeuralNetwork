#pragma once

namespace nnn {

  class FloatMatrix;

  class RowMajorFloatMatrixIterator {
   private:
    FloatMatrix* m_matrix;
    size_t m_row;
    size_t m_col;

   public:
    RowMajorFloatMatrixIterator(FloatMatrix* mat, size_t row, size_t col);
    RowMajorFloatMatrixIterator(FloatMatrix* mat);

    void Restart();

    float& Get();
    const float& Get() const;

    void Next();
    bool HasNext() const;
  };
}  // namespace nnn

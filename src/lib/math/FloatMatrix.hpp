#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>
#include <functional>

namespace nnn {

  class FloatMatrix {
   private:
    std::vector<float> m_data;
    size_t m_rows;
    size_t m_cols;
    bool m_transposed = false;
    static const unsigned int m_seed = 42;

   public:
    FloatMatrix(size_t side);
    FloatMatrix(size_t rows, size_t cols);

    static std::optional<FloatMatrix> Create(size_t rows, size_t cols, const std::vector<float>& data);
    static FloatMatrix Ones(size_t rows, size_t cols);
    static FloatMatrix Zeroes(size_t rows, size_t cols);
    static FloatMatrix Identity(size_t side);
    static FloatMatrix Random(size_t rows, size_t cols, float min = 0.0f, float max = 1.0f);

    inline size_t GetSize() const { return m_rows * m_cols; }
    inline size_t GetRowCount() const { return m_rows; }
    inline size_t GetColCount() const { return m_cols; }
    inline bool IsTransposed() const { return m_transposed; }

    inline float& operator()(size_t row, size_t col) { return m_data[ComputeIndex(row, col)]; }
    inline const float& operator()(size_t row, size_t col) const { return m_data[ComputeIndex(row, col)]; }

    std::optional<std::reference_wrapper<float>> At(size_t row, size_t col);
    bool Set(size_t row, size_t col, float value);
    void Transpose();

    float* Data();
    const float* Data() const;

    FloatMatrix operator+(const FloatMatrix& other) const;
    FloatMatrix operator*(const FloatMatrix& other) const;
    FloatMatrix operator*(float scalar) const;
    FloatMatrix& operator*=(float scalar);

    FloatMatrix Map(const std::function<float(float)>& func) const;

    std::string ToString() const;

   private:
    FloatMatrix(size_t rows, size_t cols, const std::vector<float>& data);
    FloatMatrix(size_t rows, size_t cols, float initialValue);

    inline size_t ComputeIndex(size_t row, size_t col) const {
      return (m_transposed) ? (row + m_rows * col) : (row * m_cols + col);
    }
  };

}  // namespace nnn

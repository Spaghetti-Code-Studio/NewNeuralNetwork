#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

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
    FloatMatrix(size_t rows, size_t cols, const std::vector<float>& data);
    FloatMatrix(size_t rows, size_t cols, std::vector<float>&& data);

    static std::optional<FloatMatrix> Create(size_t rows, size_t cols, const std::vector<float>& data);
    static FloatMatrix Ones(size_t rows, size_t cols);
    static FloatMatrix Zeroes(size_t rows, size_t cols);
    static FloatMatrix Identity(size_t side);
    static FloatMatrix Random(size_t rows, size_t cols, float min = 0.0f, float max = 1.0f);

    inline size_t GetSize() const { return m_rows * m_cols; }
    inline size_t GetRowCount() const { return m_rows; }
    inline size_t GetColCount() const { return m_cols; }
    inline bool IsTransposed() const { return m_transposed; }

    void Transpose();
    inline float& operator()(size_t row, size_t col) { return m_data[ComputeIndex(row, col)]; }
    inline const float& operator()(size_t row, size_t col) const { return m_data[ComputeIndex(row, col)]; }

    std::optional<float> At(size_t row, size_t col) const;
    bool Set(size_t row, size_t col, float value);

    float* Data();
    const float* Data() const;
    FloatMatrix GetColumns(size_t begin, size_t end) const;
    FloatMatrix GetColumns(const std::vector<size_t>& indices) const;

    FloatMatrix operator+(const FloatMatrix& other) const;
    FloatMatrix& operator+=(const FloatMatrix& other);
    FloatMatrix operator-(const FloatMatrix& other) const;
    FloatMatrix& operator-=(const FloatMatrix& other);

    /**
     * @brief Performs standart matrix-matrix multiplication.
     *
     * The calculation is parallelized using OpenMP if the _OPENMP macro is defined,
     * otherwise, it defaults to a serial implementation (see `MultiplySerial` method).
     *
     * @param other the right-hand side FloatMatrix in the multiplication (B in A * B)
     * @return new FloatMatrix containing the result of the matrix product
     * @throws FloatMatrixInvalidDimensionException if the number of columns in the
     * current matrix does not match the number of rows in the 'other' matrix
     */
    FloatMatrix operator*(const FloatMatrix& other) const;
    FloatMatrix MultiplySerial(const FloatMatrix& other) const;
    FloatMatrix operator*(float scalar) const;
    FloatMatrix& operator*=(float scalar);
    bool operator==(const FloatMatrix& other) const;

    FloatMatrix Map(const std::function<float(float)>& func) const;
    void MapInPlace(const std::function<float(float)>& func);
    void AddToAllCols(const FloatMatrix& vector);
    FloatMatrix Hadamard(const FloatMatrix& other) const;

    template <typename T>
    T Aggregate(const std::function<T(float)>& func) const {  //

      T result{};
      for (size_t i = 0; i < m_data.size(); ++i) {
        result += func(m_data[i]);
      }

      return result;
    }

    static FloatMatrix SumColumns(const FloatMatrix& matrix);

    std::string ToString() const;
    void Print() const;

   private:
    FloatMatrix(size_t rows, size_t cols, float initialValue);

    inline size_t ComputeIndex(size_t row, size_t col) const {
      return (m_transposed) ? (row + m_rows * col) : (row * m_cols + col);
    }
  };

}  // namespace nnn

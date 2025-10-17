#pragma once

#include <cassert>
#include <iomanip>
#include <ios>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nnn {

  class FloatMatrix {  //

   private:
    std::vector<float> m_data;
    size_t m_rows;
    size_t m_cols;
    bool m_transposed = false;
    static const unsigned int m_seed = 42;

   public:
    FloatMatrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols), m_transposed(false), m_data(rows * cols) {}
    FloatMatrix(size_t side) : m_rows(side), m_cols(side), m_transposed(false), m_data(side * side) {}

    static std::optional<FloatMatrix> Create(size_t rows, size_t cols, const std::vector<float>& data) {  //

      if (data.size() != rows * cols) {
        return {};
      }

      return FloatMatrix(rows, cols, data);
    }

    static FloatMatrix Ones(size_t rows, size_t cols) { return FloatMatrix(rows, cols, 1.0f); }

    static FloatMatrix Zeroes(size_t rows, size_t cols) { return FloatMatrix(rows, cols, 0.0f); }

    static FloatMatrix Identity(size_t side) {  //

      FloatMatrix result = Zeroes(side, side);
      for (size_t i = 0; i < side; ++i) {
        result(i, i) = 1.0f;
      }

      return result;
    }

    static FloatMatrix Random(size_t rows, size_t cols, float min = 0.0f, float max = 1.0f) {  //

      static thread_local std::mt19937 gen(m_seed);
      std::uniform_real_distribution<float> dis(min, max);

      FloatMatrix result(rows, cols);
      for (auto& val : result.m_data) {
        val = dis(gen);
      }

      return result;
    }

    inline size_t GetSize() const { return m_rows * m_cols; }

    inline size_t GetRowCount() const { return m_rows; }

    inline size_t GetColCount() const { return m_cols; }

    inline float& operator()(size_t row, size_t col) { return m_data[ComputeIndex(row, col)]; }

    inline const float& operator()(size_t row, size_t col) const { return m_data[ComputeIndex(row, col)]; }

    std::optional<std::reference_wrapper<float>> At(size_t row, size_t col) {  //

      if (row >= m_rows || col >= m_cols) {
        return std::nullopt;
      }
      return (*this)(row, col);
    }

    inline bool Set(size_t row, size_t col, float value) {  //

      if (row >= m_rows || col >= m_cols) {
        return false;
      }

      (*this)(row, col) = value;
      return true;
    }

    inline bool IsTransposed() const { return m_transposed; }

    inline void Transpose() {  //

      m_transposed = !m_transposed;
      std::swap(m_rows, m_cols);
    }

    float* Data() { return m_data.data(); }

    const float* Data() const { return m_data.data(); }

    FloatMatrix operator+(const FloatMatrix& other) const {  //

      if (m_rows != other.m_rows || m_cols != other.m_cols) {
        throw std::invalid_argument("Invalid dimensions for multiplication!");  // TODO: make custom exception
      }

      size_t side = m_rows;
      FloatMatrix result(side, side);

      for (size_t row = 0; row < side; ++row) {
        for (size_t col = 0; col < side; ++col) {
          result(row, col) = (*this)(row, col) + other(row, col);
        }
      }

      return result;
    }

    FloatMatrix operator*(const FloatMatrix& other) const {  //

      if (m_cols != other.m_rows) {
        throw std::invalid_argument("Invalid dimensions for multiplication!");  // TODO: make custom exception
      }

      FloatMatrix result = FloatMatrix(m_rows, other.m_cols);

      for (size_t i = 0; i < m_rows; ++i) {
        for (size_t j = 0; j < other.m_cols; ++j) {
          float sum = 0.0f;
          for (size_t k = 0; k < m_cols; ++k) {
            sum += (*this)(i, k) * other(k, j);
          }
          result(i, j) = sum;
        }
      }

      return result;
    }

    FloatMatrix operator*(float scalar) const {  //

      FloatMatrix result(m_rows, m_cols);
      for (size_t i = 0; i < m_data.size(); ++i) {
        result.m_data[i] = m_data[i] * scalar;
      }

      return result;
    }

    FloatMatrix& operator*=(float scalar) {  //

      for (auto& v : m_data) {
        v *= scalar;
      }

      return *this;
    }

    std::string ToString() const {  //

      std::ostringstream oss;
      oss << std::fixed << std::setprecision(3);
      oss << "FloatMatrix (" << GetRowCount() << "x" << GetColCount() << ", transposed=" << std::boolalpha
          << m_transposed << ")\n";

      for (size_t r = 0; r < GetRowCount(); ++r) {
        oss << "[ ";
        for (size_t c = 0; c < GetColCount(); ++c) {
          oss << std::setw(8) << (*this)(r, c);
          if (c + 1 < GetColCount()) oss << ", ";
        }
        oss << " ]\n";
      }

      return oss.str();
    }

   private:
    FloatMatrix(size_t rows, size_t cols, const std::vector<float>& data)
        : m_rows(rows), m_cols(cols), m_transposed(false), m_data(data) {
      assert(data.size() == rows * cols && "Data size must match matrix dimensions");  // TODO: make exception here
    }

    FloatMatrix(size_t rows, size_t cols, float initialValue)
        : m_rows(rows), m_cols(cols), m_transposed(false), m_data(rows * cols, initialValue) {}

    inline size_t ComputeIndex(size_t row, size_t col) const {
      return (m_transposed) ? (row + m_rows * col) : (row * m_cols + col);
    }
  };
}  // namespace nnn

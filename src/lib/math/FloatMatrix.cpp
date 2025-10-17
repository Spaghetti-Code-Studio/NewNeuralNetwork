#include "FloatMatrix.hpp"
#include "FloatMatrixInvalidDimensionException.hpp"

#include <iomanip>
#include <ios>
#include <random>
#include <sstream>
#include <utility>

namespace nnn {

  FloatMatrix::FloatMatrix(size_t rows, size_t cols)
      : m_rows(rows), m_cols(cols), m_transposed(false), m_data(rows * cols) {}

  FloatMatrix::FloatMatrix(size_t side) : m_rows(side), m_cols(side), m_transposed(false), m_data(side * side) {}

  FloatMatrix::FloatMatrix(size_t rows, size_t cols, const std::vector<float>& data)
      : m_rows(rows), m_cols(cols), m_transposed(false), m_data(data) {
    if (data.size() != rows * cols) {
      throw FloatMatrixInvalidDimensionException("Data size must match matrix dimensions");
    }
  }

  FloatMatrix::FloatMatrix(size_t rows, size_t cols, float initialValue)
      : m_rows(rows), m_cols(cols), m_transposed(false), m_data(rows * cols, initialValue) {}

  std::optional<FloatMatrix> FloatMatrix::Create(size_t rows, size_t cols, const std::vector<float>& data) {
    if (data.size() != rows * cols) {
      return {};
    }
    return FloatMatrix(rows, cols, data);
  }

  FloatMatrix FloatMatrix::Ones(size_t rows, size_t cols) { return FloatMatrix(rows, cols, 1.0f); }

  FloatMatrix FloatMatrix::Zeroes(size_t rows, size_t cols) { return FloatMatrix(rows, cols, 0.0f); }

  FloatMatrix FloatMatrix::Identity(size_t side) {
    FloatMatrix result = Zeroes(side, side);
    for (size_t i = 0; i < side; ++i) {
      result(i, i) = 1.0f;
    }
    return result;
  }

  FloatMatrix FloatMatrix::Random(size_t rows, size_t cols, float min, float max) {
    static thread_local std::mt19937 gen(m_seed);
    std::uniform_real_distribution<float> dis(min, max);

    FloatMatrix result(rows, cols);
    for (auto& val : result.m_data) {
      val = dis(gen);
    }
    return result;
  }

  std::optional<std::reference_wrapper<float>> FloatMatrix::At(size_t row, size_t col) {
    if (row >= m_rows || col >= m_cols) {
      return std::nullopt;
    }
    return (*this)(row, col);
  }

  bool FloatMatrix::Set(size_t row, size_t col, float value) {
    if (row >= m_rows || col >= m_cols) {
      return false;
    }
    (*this)(row, col) = value;
    return true;
  }

  void FloatMatrix::Transpose() {
    m_transposed = !m_transposed;
    std::swap(m_rows, m_cols);
  }

  float* FloatMatrix::Data() { return m_data.data(); }

  const float* FloatMatrix::Data() const { return m_data.data(); }

  FloatMatrix FloatMatrix::operator+(const FloatMatrix& other) const {
    if (m_rows != other.m_rows) {
      throw FloatMatrixInvalidDimensionException("Cannot add matrices when row count does not match.");
    } else if (m_cols != other.m_cols) {
      throw FloatMatrixInvalidDimensionException("Cannot add matrices when column count does not match.");
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

  FloatMatrix& FloatMatrix::operator+=(const FloatMatrix& other) {
    if (m_rows != other.m_rows) {
      throw FloatMatrixInvalidDimensionException("Cannot add matrices when row count does not match.");
    } else if (m_cols != other.m_cols) {
      throw FloatMatrixInvalidDimensionException("Cannot add matrices when column count does not match.");
    }

    size_t side = m_rows;
    for (size_t row = 0; row < side; ++row) {
      for (size_t col = 0; col < side; ++col) {
        (*this)(row, col) = (*this)(row, col) + other(row, col);
      }
    }
  }

  FloatMatrix FloatMatrix::operator*(const FloatMatrix& other) const {
    if (m_cols != other.m_rows) {
      throw FloatMatrixInvalidDimensionException(
          "Cannot multiply matrices when column count does not match row count.");
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

  FloatMatrix FloatMatrix::operator*(float scalar) const {
    FloatMatrix result(m_rows, m_cols);
    for (size_t i = 0; i < m_data.size(); ++i) {
      result.m_data[i] = m_data[i] * scalar;
    }
    return result;
  }

  FloatMatrix& FloatMatrix::operator*=(float scalar) {
    for (auto& v : m_data) {
      v *= scalar;
    }
    return *this;
  }

  FloatMatrix FloatMatrix::Map(const std::function<float(float)>& func) const {
    FloatMatrix result(m_rows, m_cols);
    for (size_t i = 0; i < m_data.size(); ++i) {
      result.m_data[i] = func(m_data[i]);
    }
    return result;
  }

  std::string FloatMatrix::ToString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "FloatMatrix (" << GetRowCount() << "x" << GetColCount() << ", transposed=" << std::boolalpha << m_transposed
        << ")\n";

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
}  // namespace nnn

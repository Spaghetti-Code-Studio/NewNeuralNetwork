#include <cmath>
#include <stdexcept> 
#include <utility>

#include "FloatMatrix.hpp"
#include "Softmax.hpp"

static inline float FindMaxInColumn(const nnn::FloatMatrix& matrix, size_t col) {  //

  float max = matrix(0, col);
  for (size_t row = 1; row < matrix.GetRowCount(); row++) {
    max = std::max(max, matrix(row, col));
  }

  return max;
}

static inline float SumExponentialsInColumn(nnn::FloatMatrix& matrix, size_t col, float maxValueInColumn) {  //

  float sum = 0.0f;
  for (size_t row = 0; row < matrix.GetRowCount(); row++) {
    matrix(row, col) = std::exp(matrix(row, col) - maxValueInColumn);
    sum += matrix(row, col);
  }

  return sum;
}

namespace nnn {

  void Softmax::Evaluate(FloatMatrix& input) const {  //

    for (size_t col = 0; col < input.GetColCount(); ++col) {  //

      float maxValueInColumn = FindMaxInColumn(input, col);
      float expSum = SumExponentialsInColumn(input, col, maxValueInColumn);

      // TODO: leave it here, change it? I am not sure if this case can occur or no...
      if (expSum == 0.0f) {
        throw std::runtime_error("Numerical error in Softmax (division by zero)!");
      }

      float recip = 1.0f / expSum;
      for (size_t row = 0; row < input.GetRowCount(); row++) {
        input(row, col) *= recip;
      }
    }
  }

  void Softmax::Derivative(FloatMatrix& input) const {  //

    // TODO: implement this
    throw std::runtime_error("Not implemented yet!");
  }
}  // namespace nnn

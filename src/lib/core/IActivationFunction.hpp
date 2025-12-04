#pragma once

#include "FloatMatrix.hpp"

namespace nnn {

  /**
   * @brief The interface for a general activation function.
   */
  class IActivationFunction {
   public:
    virtual ~IActivationFunction() = 0;

    /**
     * @brief In-place evaluation of the activation function for the given input.   
     */
    virtual void Evaluate(FloatMatrix& input) const = 0;

    /**
     * @brief In-place evaluation of the derivate for the given input.   
     */
    virtual void Derivative(FloatMatrix& input) const = 0;
  };

  inline IActivationFunction::~IActivationFunction() = default;
}  // namespace nnn

#pragma once
#include <exception>
#include <string>

namespace nnn {

  class FloatMatrixInvalidDimensionException : public std::exception {
   private:
    std::string message;

   public:
    FloatMatrixInvalidDimensionException(const char* msg) : message(msg) {}

    const char* what() const noexcept { return message.c_str(); }
  };
}  // namespace nnn
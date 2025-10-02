#pragma once

#include <string>

#include <result.hpp>

namespace nnn::math {
  void Print();
  cpp::result<int, std::string> DoMath(int parameter);
}  // namespace nnn::math

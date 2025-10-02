#include <iostream>

#include "Math.hpp"

void nnn::math::Print() { std::cout << "Hello world!" << std::endl; }

cpp::result<int, std::string> nnn::math::DoMath(int parameter) {
  if (parameter >= 0) {
    return 42;
  }
  return cpp::fail("Number is not Cizek number!");
}

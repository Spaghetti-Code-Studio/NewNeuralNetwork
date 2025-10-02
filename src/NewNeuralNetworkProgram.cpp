#include <iostream>
#include <string>

#include "Math.hpp"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "Usage: \n" << argv[0] << " <number>" << std::endl;
    return -1;
  }

  int parameter;
  try {
    parameter = std::stoi(argv[1]);
  } catch (const std::invalid_argument& e) {
    std::cerr << "[IO ERROR]: Invalid number format!\n";
    return -1;
  } catch (const std::out_of_range& e) {
    std::cerr << "[IO ERROR]: Number out of range!\n";
    return -1;
  }

  nnn::math::Print();
  auto result = nnn::math::DoMath(parameter);
  if (result.has_error()) {
    std::cerr << "[MATH ERROR]: <" << result.error() << ">!" << std::endl;
    return -1;
  }
  std::cout << "Result is " << result.value() << "." << std::endl;
  return 0;
}

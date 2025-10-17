#include <iostream>

#include <FloatMatrix.hpp>

int main(int argc, char* argv[]) {
  if (argc != 1) {
    std::cout << "Usage: \n" << argv[0] << std::endl;
    return -1;
  }

  auto random = nnn::FloatMatrix::Random(6, 9);
  std::cout << random.ToString() << std::endl;
  random.Transpose();
  std::cout << random.ToString() << std::endl;

  return 0;
}

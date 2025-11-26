#include <iostream>

#include <CSVReader.hpp>
#include <FloatMatrix.hpp>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "Usage: \n" << argv[0] << " <filepath to CSV file with matrix>" << std::endl;
    return -1;
  }

  auto reader = nnn::CSVReader();
  auto readResult = reader.Read(argv[1]);
  if (readResult.has_error()) {
    std::cout << readResult.error() << std::endl;
    return -1;
  }

  auto matrix = readResult.value();
  matrix->Print();
  matrix->Transpose();
  matrix->Print();

  return 0;
}

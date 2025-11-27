#include <chrono>
#include <ios>
#include <iostream>
#include <string>

#include <CSVReader.hpp>
#include <FloatMatrix.hpp>
#include <Timer.hpp>

int main(int argc, char* argv[]) {  //

  if (argc != 3) {
    std::cout << "Usage: \n"
              << argv[0]
              << " <filepath to CSV file with matrix> "
                 "<true/false flag if the loaded matrix should be printed>"
              << std::endl;
    return -1;
  }

  nnn::Timer timer;
  timer.Start();

  auto reader = nnn::CSVReader();
  auto readResult = reader.Read(argv[1]);

  nnn::Timer::Second readTime = timer.End();

  if (readResult.has_error()) {
    std::cout << readResult.error() << std::endl;
    return -1;
  }

  std::cout << "Reading of CSV file took " << readTime << " seconds.\n" << std::endl;
  auto matrix = readResult.value();

  std::string print_flag_arg = argv[2];
  if (print_flag_arg == "true" || print_flag_arg == "TRUE") {
    matrix->Print();
    matrix->Transpose();
    matrix->Print();
  } else {
    std::cout << "FloatMatrix (" << matrix->GetRowCount() << "x" << matrix->GetColCount()
              << ", transposed=" << std::boolalpha << matrix->IsTransposed() << ")\n";
  }

  return 0;
}

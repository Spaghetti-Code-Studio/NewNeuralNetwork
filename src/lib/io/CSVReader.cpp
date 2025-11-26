#include "CSVReader.hpp"

#include <exception>
#include <filesystem>
#include <fstream>
#include <iosfwd>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

cpp::result<std::shared_ptr<nnn::FloatMatrix>, std::string> nnn::CSVReader::Read(std::filesystem::path filepath) {  //

  std::ifstream inputFile;
  inputFile.open(filepath);

  if (!inputFile.is_open()) {
    try {
      std::filesystem::path absolute_filepath = std::filesystem::absolute(filepath);
      return cpp::fail("File <" + absolute_filepath.string() + "> was not found or access denied.");
    } catch (const std::filesystem::filesystem_error& e) {
      return cpp::fail("Error resolving path: <" + filepath.string() + ">. Details: " + e.what());
    }
  }

  std::vector<float> rawData;
  int rows = 0;
  int cols = 0;

  std::string line;
  while (std::getline(inputFile, line)) {  //

    if (line.empty()) {
      continue;
    }

    std::stringstream ss(line);
    std::string cell;
    int current_cols = 0;

    while (std::getline(ss, cell, m_delimiter)) {
      try {
        float value = std::stof(cell);
        rawData.push_back(value);
        current_cols++;
      } catch (const std::exception& e) {
        inputFile.close();
        std::string error_msg = "Error while parsing float in file at line <" + std::to_string(rows + 1) +
                                ">, cell value: <" + cell + ">! Details: <" + e.what() + ">.";
        return cpp::fail(error_msg);
      }
    }

    if (rows == 0) {
      cols = current_cols;
    } else if (current_cols != cols) {
      inputFile.close();
      std::string error_msg = "Inconsistent column count at line <" + std::to_string(rows + 1) + ">. Found <" +
                              std::to_string(current_cols) + "> columns, but expected <" + std::to_string(cols) + ">!";
      return cpp::fail(error_msg);
    }

    rows++;
  }

  inputFile.close();
  return std::make_shared<nnn::FloatMatrix>(rows, cols, std::move(rawData));
}

bool nnn::CSVReader::IsValid(
    cpp::result<std::shared_ptr<FloatMatrix>, std::string> result, size_t expectedCols, size_t expectedRows) {  //

  if (result.has_error()) {
    return false;
  }

  return result.value()->GetColCount() == expectedCols && result.value()->GetRowCount() == expectedRows;
}

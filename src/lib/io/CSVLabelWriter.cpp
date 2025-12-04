#include "CSVLabelWriter.hpp"

#include <fstream>
#include <stdexcept>

namespace nnn {

  cpp::result<void, IoError> CSVLabelsWriter::Write(std::filesystem::path filepath, const FloatMatrix& data) {  //

    std::ofstream outputFile;
    outputFile.open(filepath, std::ios::out);

    if (!outputFile.is_open()) {
      try {
        std::filesystem::path absolute_filepath = std::filesystem::absolute(filepath);
        return cpp::fail("File <" + absolute_filepath.string() + "> failed to open for writing.");
      } catch (const std::filesystem::filesystem_error& e) {
        return cpp::fail("Error resolving path: <" + filepath.string() + ">. Details: " + e.what());
      }
    }

    try {
      const size_t classes = data.GetRowCount();
      const size_t vectors = data.GetColCount();

      // TODO: code duplication with the logic in TestDataSoftmaxEvaluator::Evaluate() method.
      for (int col = 0; col < vectors; ++col) {  //

        float max = -std::numeric_limits<float>::infinity();
        int max_index = -1;

        for (int row = 0; row < classes; ++row) {
          float current_probability = data(row, col);
          if (current_probability > max) {
            max = current_probability;
            max_index = row;
          }
        }

        outputFile << max_index << '\n';
      }

      if (outputFile.fail()) {
        return cpp::fail("I/O error occurred during matrix writing!");
      }

    } catch (const std::exception& e) {
      return cpp::fail(std::string("Internal error during writing: ") + e.what());
    }

    outputFile.close();

    return {};
  }
}  // namespace nnn
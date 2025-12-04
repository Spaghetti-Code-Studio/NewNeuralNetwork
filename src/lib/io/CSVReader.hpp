#pragma once

#include "IReader.hpp"

namespace nnn {

  class CSVReader : public IReader {
   public:
    CSVReader(char delimiter = ',') : m_delimiter(delimiter) {}
    cpp::result<std::shared_ptr<FloatMatrix>, IoError> Read(std::filesystem::path filepath) override;
    static bool IsValid(
        cpp::result<std::shared_ptr<FloatMatrix>, IoError> result, size_t expectedCols, size_t expectedRows);

   private:
    char m_delimiter;
  };

}  // namespace nnn

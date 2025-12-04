#pragma once

#include "IWriter.hpp"

namespace nnn {

  class CSVLabelsWriter : public IWriter {
   public:
    CSVLabelsWriter() = default;
    cpp::result<void, IoError> Write(std::filesystem::path filepath, const FloatMatrix& data) override;
  };

}  // namespace nnn

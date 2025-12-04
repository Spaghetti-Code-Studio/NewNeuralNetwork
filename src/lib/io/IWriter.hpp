#pragma once

#include <filesystem>

#include "result.hpp"

#include "FloatMatrix.hpp"
#include "IoError.hpp"

namespace nnn {
  /**
   * @brief Interface for generic float matrix writer.
   */
  class IWriter {
   public:
    virtual ~IWriter() = 0;
    virtual cpp::result<void, IoError> Write(std::filesystem::path filepath, const FloatMatrix& data) = 0;
  };

  inline IWriter::~IWriter() = default;
}  // namespace nnn

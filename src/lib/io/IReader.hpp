#pragma once

#include <filesystem>

#include "result.hpp"

#include "IoError.hpp"
#include "FloatMatrix.hpp"

namespace nnn {
  /**
   * @brief Interface for generic float matrix reader.
   */
  class IReader {
   public:
    virtual ~IReader() = 0;
    virtual cpp::result<std::shared_ptr<FloatMatrix>, IoError> Read(std::filesystem::path filepath) = 0;
  };

  inline IReader::~IReader() = default;
}  // namespace nnn

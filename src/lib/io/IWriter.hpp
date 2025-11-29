#pragma once

#include <filesystem>
#include <string>

#include "result.hpp"

#include "FloatMatrix.hpp"

namespace nnn {

  class IWriter {
   public:
    virtual ~IWriter() = 0;
    virtual cpp::result<void, std::string> Write(std::filesystem::path filepath, const FloatMatrix& data) = 0;
  };

  inline IWriter::~IWriter() = default;
}  // namespace nnn

#pragma once

#include <filesystem>
#include <string>

#include "result.hpp"

#include "FloatMatrix.hpp"

namespace nnn {

  class IReader {
   public:
    virtual ~IReader() = 0;
    virtual cpp::result<std::shared_ptr<FloatMatrix>, std::string> Read(std::filesystem::path filepath) = 0;
  };

  inline IReader::~IReader() = default;
}  // namespace nnn

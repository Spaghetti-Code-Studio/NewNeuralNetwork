#pragma once

#include <chrono>

namespace nnn {

  class Timer {
   public:
    using Second = double;
    Timer() = default;

    void Start() { m_startTime = std::chrono::high_resolution_clock::now(); }

    Second End() {  //

      auto timeSpan = std::chrono::duration<Second>(std::chrono::high_resolution_clock::now() - m_startTime).count();
      m_startTime = std::chrono::high_resolution_clock::time_point();

      return timeSpan;
    }

   private:
    std::chrono::high_resolution_clock::time_point m_startTime;
  };
}  // namespace nnn
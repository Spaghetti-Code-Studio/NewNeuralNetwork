#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>

// This file expects OpemMP installed.
#include <omp.h>

#include "FloatMatrix.hpp"

TEST_CASE("Matrix multiplication performance") {
  auto a = nnn::FloatMatrix::Random(784, 176, -1.0f, 1.0f);
  auto b = nnn::FloatMatrix::Random(176, 64, -1.0f, 1.0f);

  int num_threads = omp_get_max_threads();
  std::cout << "Max threads available: " << num_threads << std::endl;

  BENCHMARK("Parallel A * B (Max Threads Available)") { return a * b; };

  BENCHMARK("Serial A * B (Single Thread)") { return a.MultiplySerial(b); };

  BENCHMARK("Parallel A * B (4 Threads)") {
    omp_set_num_threads(4);
    return a * b;
  };
}

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test") {
  int x = 5;
  int y = 6;
  int result = x - y;
  CHECK(result == -1);
}

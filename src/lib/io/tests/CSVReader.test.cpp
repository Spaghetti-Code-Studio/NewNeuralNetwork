#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "CSVReader.hpp"
#include "FloatMatrix.hpp"

TEST_CASE("Valid file") {
  nnn::CSVReader reader;
  auto readResult = reader.Read("../../../../../src/lib/io/tests/testData.csv");
  REQUIRE(readResult.has_value());

  auto matrix = readResult.value();
  CHECK_THAT((*matrix)(0, 0), Catch::Matchers::WithinAbs(8.589f, 0.001));
  CHECK_THAT((*matrix)(0, 1), Catch::Matchers::WithinAbs(48.0f, 0.001));
  CHECK_THAT((*matrix)(0, 2), Catch::Matchers::WithinAbs(-85.87f, 0.001));
  CHECK_THAT((*matrix)(0, 3), Catch::Matchers::WithinAbs(1.00001f, 0.001));

  CHECK_THAT((*matrix)(1, 0), Catch::Matchers::WithinAbs(789568.589f, 0.001));
  CHECK_THAT((*matrix)(1, 1), Catch::Matchers::WithinAbs(48.0f, 0.001));
  CHECK_THAT((*matrix)(1, 2), Catch::Matchers::WithinAbs(-85.0f, 0.001));
  CHECK_THAT((*matrix)(1, 3), Catch::Matchers::WithinAbs(0.0f, 0.001));
}

TEST_CASE("Valid file - different delimiter") {
  nnn::CSVReader reader(';');
  auto readResult = reader.Read("../../../../../src/lib/io/tests/testDataSemicolon.csv");
  REQUIRE(readResult.has_value());

  auto matrix = readResult.value();
  CHECK_THAT((*matrix)(0, 0), Catch::Matchers::WithinAbs(8.589f, 0.001));
  CHECK_THAT((*matrix)(0, 1), Catch::Matchers::WithinAbs(48.0f, 0.001));
  CHECK_THAT((*matrix)(0, 2), Catch::Matchers::WithinAbs(-85.87f, 0.001));
  CHECK_THAT((*matrix)(0, 3), Catch::Matchers::WithinAbs(1.00001f, 0.001));

  CHECK_THAT((*matrix)(1, 0), Catch::Matchers::WithinAbs(789568.589f, 0.001));
  CHECK_THAT((*matrix)(1, 1), Catch::Matchers::WithinAbs(48.0f, 0.001));
  CHECK_THAT((*matrix)(1, 2), Catch::Matchers::WithinAbs(-85.0f, 0.001));
  CHECK_THAT((*matrix)(1, 3), Catch::Matchers::WithinAbs(0.0f, 0.001));
}

TEST_CASE("Valid file - empty") {
  nnn::CSVReader reader;
  auto readResult = reader.Read("../../../../../src/lib/io/tests/empty.csv");
  REQUIRE(readResult.has_value());

  auto matrix = readResult.value();
  CHECK(matrix->GetColCount() == 0);
  CHECK(matrix->GetRowCount() == 0);
}

TEST_CASE("Invalid file - corrupted data") {
  nnn::CSVReader reader;
  auto readResult = reader.Read("../../../../../src/lib/io/tests/testDataInvalid.csv");
  REQUIRE(readResult.has_error());
}

TEST_CASE("Invalid file - unknown file") {
  nnn::CSVReader reader;
  auto readResult = reader.Read("../../../../../src/lib/io/tests/UNKNOWN.csv");
  REQUIRE(readResult.has_error());
}
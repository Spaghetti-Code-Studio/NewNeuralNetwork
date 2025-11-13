#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ColumnMajorFloatMatrixIterator.hpp"
#include "FloatMatrix.hpp"
#include "RowMajorFloatMatrixIterator.hpp"

TEST_CASE("Basic rectangle matrix getters") {
  size_t row = 7;
  size_t col = 8;
  nnn::FloatMatrix rectangle(row, col);
  CHECK(rectangle.GetSize() == row * col);
  CHECK(rectangle.GetColCount() == col);
  CHECK(rectangle.GetRowCount() == row);

  for (size_t r = 0; r < row; ++r) {
    for (size_t c = 0; c < col; ++c) {
      CHECK(rectangle(r, c) == 0.0f);
    }
  }

  CHECK_FALSE(rectangle.At(row, col).has_value());
  CHECK(rectangle.At(row - 1, col - 1).value() == 0.0f);
}

TEST_CASE("Matrix creation") {
  auto rectangleResult = nnn::FloatMatrix::Create(2, 3, {1.0f, 2.0f, 3.0f, 4.0f});
  REQUIRE_FALSE(rectangleResult.has_value());

  rectangleResult = nnn::FloatMatrix::Create(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  REQUIRE(rectangleResult.has_value());

  auto ones = nnn::FloatMatrix::Ones(2, 2);
  std::vector expected = {1.0f, 1.0f, 1.0f, 1.0f};

  auto data = ones.Data();
  for (size_t i = 0; i < ones.GetSize(); ++i) {
    CHECK(data[i] == expected[i]);
  }

  // ------------------------------------------------

  auto zeroes = nnn::FloatMatrix::Zeroes(2, 2);
  expected = {0.0f, 0.0f, 0.0f, 0.0f};

  data = zeroes.Data();
  for (size_t i = 0; i < zeroes.GetSize(); ++i) {
    CHECK(data[i] == expected[i]);
  }

  // ------------------------------------------------

  auto identity = nnn::FloatMatrix::Identity(3);
  expected = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};

  data = identity.Data();
  for (size_t i = 0; i < identity.GetSize(); ++i) {
    CHECK(data[i] == expected[i]);
  }

  // ------------------------------------------------

  auto random = nnn::FloatMatrix::Random(2, 2);
  expected = {0.375f, 0.797f, 0.951f, 0.183f};

  data = random.Data();
  for (size_t i = 0; i < random.GetSize(); ++i) {
    CHECK_THAT(data[i], Catch::Matchers::WithinAbs(expected[i], 0.001));
  }
}

TEST_CASE("Basic rectangle matrix iterators") {
  {
    auto rectangleResult = nnn::FloatMatrix::Create(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    REQUIRE(rectangleResult.has_value());
    auto& rectangle = rectangleResult.value();

    nnn::RowMajorFloatMatrixIterator rowIterator = nnn::RowMajorFloatMatrixIterator(&rectangle);

    std::vector rowData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto dataIterator = rowData.begin();

    while (rowIterator.HasNext() && dataIterator != rowData.end()) {
      CHECK_THAT(rowIterator.Get(), Catch::Matchers::WithinAbs(*dataIterator, 0.001));
      rowIterator.Next();
      ++dataIterator;
    }

    CHECK_FALSE(rowIterator.HasNext());
    CHECK(dataIterator == rowData.end());
  }
  {
    auto rectangleResult = nnn::FloatMatrix::Create(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    REQUIRE(rectangleResult.has_value());
    auto& rectangle = rectangleResult.value();

    nnn::ColumnMajorFloatMatrixIterator colIterator = nnn::ColumnMajorFloatMatrixIterator(&rectangle);

    std::vector colData = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    auto dataIterator = colData.begin();

    while (colIterator.HasNext() && dataIterator != colData.end()) {
      CHECK_THAT(colIterator.Get(), Catch::Matchers::WithinAbs(*dataIterator, 0.001));
      colIterator.Next();
      ++dataIterator;
    }

    CHECK_FALSE(colIterator.HasNext());
    CHECK(dataIterator == colData.end());
  }
}

TEST_CASE("Basic matrix transposed") {
  {
    auto rectangleResult = nnn::FloatMatrix::Create(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    REQUIRE(rectangleResult.has_value());
    auto& rectangle = rectangleResult.value();

    CHECK(rectangle.GetRowCount() == 2);
    CHECK(rectangle.GetColCount() == 3);
    CHECK_FALSE(rectangle.IsTransposed());

    CHECK(rectangle(0, 0) == 1.0f);
    CHECK(rectangle(0, 1) == 2.0f);
    CHECK(rectangle(0, 2) == 3.0f);
    CHECK(rectangle(1, 0) == 4.0f);
    CHECK(rectangle(1, 1) == 5.0f);
    CHECK(rectangle(1, 2) == 6.0f);
    CHECK(rectangle.At(0, 0).value() == 1.0f);
    CHECK(rectangle.At(0, 1).value() == 2.0f);
    CHECK(rectangle.At(0, 2).value() == 3.0f);
    CHECK(rectangle.At(1, 0).value() == 4.0f);
    CHECK(rectangle.At(1, 1).value() == 5.0f);
    CHECK(rectangle.At(1, 2).value() == 6.0f);

    // --------------------------------------------

    rectangle.Transpose();
    CHECK(rectangle.GetRowCount() == 3);
    CHECK(rectangle.GetColCount() == 2);
    CHECK(rectangle.IsTransposed());

    CHECK(rectangle(0, 0) == 1.0f);
    CHECK(rectangle(0, 1) == 4.0f);
    CHECK(rectangle(1, 0) == 2.0f);
    CHECK(rectangle(1, 1) == 5.0f);
    CHECK(rectangle(2, 0) == 3.0f);
    CHECK(rectangle(2, 1) == 6.0f);
    CHECK(rectangle.At(0, 0).value() == 1.0f);
    CHECK(rectangle.At(0, 1).value() == 4.0f);
    CHECK(rectangle.At(1, 0).value() == 2.0f);
    CHECK(rectangle.At(1, 1).value() == 5.0f);
    CHECK(rectangle.At(2, 0).value() == 3.0f);
    CHECK(rectangle.At(2, 1).value() == 6.0f);

    // --------------------------------------------

    rectangle.Transpose();
    CHECK(rectangle.GetRowCount() == 2);
    CHECK(rectangle.GetColCount() == 3);
    CHECK_FALSE(rectangle.IsTransposed());

    CHECK(rectangle(0, 0) == 1.0f);
    CHECK(rectangle(0, 1) == 2.0f);
    CHECK(rectangle(0, 2) == 3.0f);
    CHECK(rectangle(1, 0) == 4.0f);
    CHECK(rectangle(1, 1) == 5.0f);
    CHECK(rectangle(1, 2) == 6.0f);
    CHECK(rectangle.At(0, 0).value() == 1.0f);
    CHECK(rectangle.At(0, 1).value() == 2.0f);
    CHECK(rectangle.At(0, 2).value() == 3.0f);
    CHECK(rectangle.At(1, 0).value() == 4.0f);
    CHECK(rectangle.At(1, 1).value() == 5.0f);
    CHECK(rectangle.At(1, 2).value() == 6.0f);

    // --------------------------------------------

    rectangle.Transpose();
    CHECK(rectangle.GetRowCount() == 3);
    CHECK(rectangle.GetColCount() == 2);
    CHECK(rectangle.IsTransposed());

    CHECK(rectangle(0, 0) == 1.0f);
    CHECK(rectangle(0, 1) == 4.0f);
    CHECK(rectangle(1, 0) == 2.0f);
    CHECK(rectangle(1, 1) == 5.0f);
    CHECK(rectangle(2, 0) == 3.0f);
    CHECK(rectangle(2, 1) == 6.0f);
    CHECK(rectangle.At(0, 0).value() == 1.0f);
    CHECK(rectangle.At(0, 1).value() == 4.0f);
    CHECK(rectangle.At(1, 0).value() == 2.0f);
    CHECK(rectangle.At(1, 1).value() == 5.0f);
    CHECK(rectangle.At(2, 0).value() == 3.0f);
    CHECK(rectangle.At(2, 1).value() == 6.0f);
  }

  {
    auto rectangleResult = nnn::FloatMatrix::Create(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    REQUIRE(rectangleResult.has_value());
    auto& rectangle = rectangleResult.value();

    CHECK(rectangle.GetRowCount() == 2);
    CHECK(rectangle.GetColCount() == 2);
    CHECK_FALSE(rectangle.IsTransposed());

    CHECK(rectangle(0, 0) == 1.0f);
    CHECK(rectangle(0, 1) == 2.0f);
    CHECK(rectangle(1, 0) == 3.0f);
    CHECK(rectangle(1, 1) == 4.0f);
    CHECK(rectangle.At(0, 0).value() == 1.0f);
    CHECK(rectangle.At(0, 1).value() == 2.0f);
    CHECK(rectangle.At(1, 0).value() == 3.0f);
    CHECK(rectangle.At(1, 1).value() == 4.0f);

    // --------------------------------------------

    rectangle.Transpose();
    CHECK(rectangle.GetRowCount() == 2);
    CHECK(rectangle.GetColCount() == 2);
    CHECK(rectangle.IsTransposed());

    CHECK(rectangle(0, 0) == 1.0f);
    CHECK(rectangle(0, 1) == 3.0f);
    CHECK(rectangle(1, 0) == 2.0f);
    CHECK(rectangle(1, 1) == 4.0f);
    CHECK(rectangle.At(0, 0).value() == 1.0f);
    CHECK(rectangle.At(0, 1).value() == 3.0f);
    CHECK(rectangle.At(1, 0).value() == 2.0f);
    CHECK(rectangle.At(1, 1).value() == 4.0f);
  }

  {
    auto rectangleResult = nnn::FloatMatrix::Create(2, 2, {6.0f, 9.0f, 9.0f, 12.0f});
    REQUIRE(rectangleResult.has_value());
    auto& rectangle = rectangleResult.value();

    CHECK(rectangle.GetRowCount() == 2);
    CHECK(rectangle.GetColCount() == 2);
    CHECK_FALSE(rectangle.IsTransposed());

    CHECK(rectangle(0, 0) == 6.0f);
    CHECK(rectangle(0, 1) == 9.0f);
    CHECK(rectangle(1, 0) == 9.0f);
    CHECK(rectangle(1, 1) == 12.0f);
    CHECK(rectangle.At(0, 0).value() == 6.0f);
    CHECK(rectangle.At(0, 1).value() == 9.0f);
    CHECK(rectangle.At(1, 0).value() == 9.0f);
    CHECK(rectangle.At(1, 1).value() == 12.0f);

    // --------------------------------------------

    rectangle.Transpose();
    CHECK(rectangle.GetRowCount() == 2);
    CHECK(rectangle.GetColCount() == 2);
    CHECK(rectangle.IsTransposed());

    CHECK(rectangle(0, 0) == 6.0f);
    CHECK(rectangle(0, 1) == 9.0f);
    CHECK(rectangle(1, 0) == 9.0f);
    CHECK(rectangle(1, 1) == 12.0f);
    CHECK(rectangle.At(0, 0).value() == 6.0f);
    CHECK(rectangle.At(0, 1).value() == 9.0f);
    CHECK(rectangle.At(1, 0).value() == 9.0f);
    CHECK(rectangle.At(1, 1).value() == 12.0f);
  }
}

TEST_CASE("Basic square matrix addition") {
  auto a = nnn::FloatMatrix::Create(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
  auto b = nnn::FloatMatrix::Create(2, 2, {5.0f, 6.0f, 7.0f, 8.0f});

  REQUIRE(a.has_value());
  REQUIRE(b.has_value());

  auto result = a.value() + b.value();

  CHECK(result(0, 0) == 6.0f);
  CHECK(result(0, 1) == 8.0f);
  CHECK(result(1, 0) == 10.0f);
  CHECK(result(1, 1) == 12.0f);

  result += b.value();
  CHECK(result(0, 0) == 11.0f);
  CHECK(result(0, 1) == 14.0f);
  CHECK(result(1, 0) == 17.0f);
  CHECK(result(1, 1) == 20.0f);
}

TEST_CASE("Transposed square matrix addition") {
  auto a_result = nnn::FloatMatrix::Create(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
  auto b_result = nnn::FloatMatrix::Create(2, 2, {5.0f, 6.0f, 7.0f, 8.0f});

  REQUIRE(a_result.has_value());
  REQUIRE(b_result.has_value());

  auto& a = a_result.value();
  auto& b = b_result.value();

  // ---------------------------------------------

  a.Transpose();
  auto ab = a + b;

  CHECK(ab(0, 0) == 6.0f);
  CHECK(ab(0, 1) == 9.0f);
  CHECK(ab(1, 0) == 9.0f);
  CHECK(ab(1, 1) == 12.0f);

  b.Transpose();
  ab = a + b;

  CHECK(ab(0, 0) == 6.0f);
  CHECK(ab(0, 1) == 10.0f);
  CHECK(ab(1, 0) == 8.0f);
  CHECK(ab(1, 1) == 12.0f);

  a.Transpose();
  ab = a + b;

  CHECK(ab(0, 0) == 6.0f);
  CHECK(ab(0, 1) == 9.0f);
  CHECK(ab(1, 0) == 9.0f);
  CHECK(ab(1, 1) == 12.0f);
}

TEST_CASE("Basic matrix multiplication") {
  auto a = nnn::FloatMatrix::Create(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
  auto b = nnn::FloatMatrix::Create(2, 2, {5.0f, 6.0f, 7.0f, 8.0f});

  REQUIRE(a.has_value());
  REQUIRE(b.has_value());

  auto ab = a.value() * b.value();

  CHECK(ab(0, 0) == 19.0f);
  CHECK(ab(0, 1) == 22.0f);
  CHECK(ab(1, 0) == 43.0f);
  CHECK(ab(1, 1) == 50.0f);

  // --------------------------------------------

  auto c = nnn::FloatMatrix::Create(2, 2, {3.1f, -4.9f, -2.6f, 5.3f});
  auto d = nnn::FloatMatrix::Create(2, 3, {8.11f, -2.0f, 0.0f, 0.0f, 1.75f, 10.2f});

  REQUIRE(c.has_value());
  REQUIRE(d.has_value());

  auto cd = c.value() * d.value();

  CHECK_THAT(cd(0, 0), Catch::Matchers::WithinAbs(25.141f, 0.001));
  CHECK_THAT(cd(0, 1), Catch::Matchers::WithinAbs(-14.775f, 0.001));
  CHECK_THAT(cd(0, 2), Catch::Matchers::WithinAbs(-49.98f, 0.001));
  CHECK_THAT(cd(1, 0), Catch::Matchers::WithinAbs(-21.086f, 0.001));
  CHECK_THAT(cd(1, 1), Catch::Matchers::WithinAbs(14.475f, 0.001));
  CHECK_THAT(cd(1, 2), Catch::Matchers::WithinAbs(54.06f, 0.001));

  // --------------------------------------------

  d.value().Transpose();
  REQUIRE_THROWS(c.value() * d.value());
}

TEST_CASE("Transposed square matrix multiplication") {
  auto a = nnn::FloatMatrix::Create(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
  auto b = nnn::FloatMatrix::Create(2, 2, {5.0f, 6.0f, 7.0f, 8.0f});

  REQUIRE(a.has_value());
  REQUIRE(b.has_value());

  b.value().Transpose();

  auto ab = a.value() * b.value();

  CHECK(ab(0, 0) == 17.0f);
  CHECK(ab(0, 1) == 23.0f);
  CHECK(ab(1, 0) == 39.0f);
  CHECK(ab(1, 1) == 53.0f);

  a.value().Transpose();

  ab = a.value() * b.value();

  CHECK(ab(0, 0) == 23.0f);
  CHECK(ab(0, 1) == 31.0f);
  CHECK(ab(1, 0) == 34.0f);
  CHECK(ab(1, 1) == 46.0f);

  b.value().Transpose();

  ab = a.value() * b.value();

  CHECK(ab(0, 0) == 26.0f);
  CHECK(ab(0, 1) == 30.0f);
  CHECK(ab(1, 0) == 38.0f);
  CHECK(ab(1, 1) == 44.0f);
}

TEST_CASE("Transposed rectangle matrix multiplication") {
  auto aResult = nnn::FloatMatrix::Create(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto bResult = nnn::FloatMatrix::Create(3, 2, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});

  REQUIRE(aResult.has_value());
  REQUIRE(bResult.has_value());

  auto& a = aResult.value();
  auto& b = bResult.value();

  b.Transpose();
  REQUIRE_THROWS(a * b);

  b.Transpose();
  REQUIRE_NOTHROW(a * b);

  auto ab = a * b;
  CHECK(ab.GetColCount() == 2);
  CHECK(ab.GetRowCount() == 2);

  CHECK(ab(0, 0) == 220.0f);
  CHECK(ab(0, 1) == 280.0f);
  CHECK(ab(1, 0) == 490.0f);
  CHECK(ab(1, 1) == 640.0f);

  a.Transpose();
  b.Transpose();

  ab = a * b;
  CHECK(ab.GetColCount() == 3);
  CHECK(ab.GetRowCount() == 3);

  CHECK(ab(0, 0) == 90.0f);
  CHECK(ab(0, 1) == 190.0f);
  CHECK(ab(0, 2) == 290.0f);
  CHECK(ab(1, 0) == 120.0f);
  CHECK(ab(1, 1) == 260.0f);
  CHECK(ab(1, 2) == 400.0f);
  CHECK(ab(2, 0) == 150.0f);
  CHECK(ab(2, 1) == 330.0f);
  CHECK(ab(2, 2) == 510.0f);
}

TEST_CASE("Basic rectangle matrix scalar multiplication") {
  auto aResult = nnn::FloatMatrix::Create(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto bResult = nnn::FloatMatrix::Create(2, 3, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});

  REQUIRE(aResult.has_value());
  REQUIRE(bResult.has_value());

  auto& a = aResult.value();
  auto& b = bResult.value();

  nnn::FloatMatrix m = a * 0.5f;
  CHECK(m(0, 0) == 0.5f);
  CHECK(m(0, 1) == 1.0f);
  CHECK(m(0, 2) == 1.5f);
  CHECK(m(1, 0) == 2.0f);
  CHECK(m(1, 1) == 2.5f);
  CHECK(m(1, 2) == 3.0f);

  b *= 0.5f;
  CHECK(b(0, 0) == 5.0f);
  CHECK(b(0, 1) == 10.0f);
  CHECK(b(0, 2) == 15.0f);
  CHECK(b(1, 0) == 20.0f);
  CHECK(b(1, 1) == 25.0f);
  CHECK(b(1, 2) == 30.0f);
}

TEST_CASE("Function mapping and aggregating to basic matrix") {
  auto aResult = nnn::FloatMatrix::Create(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
  REQUIRE(aResult.has_value());
  auto& a = aResult.value();

  // Test squaring elements
  nnn::FloatMatrix squared = a.Map([](float x) { return x * x; });
  CHECK(squared(0, 0) == 1.0f);
  CHECK(squared(0, 1) == 4.0f);
  CHECK(squared(1, 0) == 9.0f);
  CHECK(squared(1, 1) == 16.0f);

  squared.MapInPlace([](float x) { return x - 16.0f; });
  CHECK(squared(0, 0) == -15.0f);
  CHECK(squared(0, 1) == -12.0f);
  CHECK(squared(1, 0) == -7.0f);
  CHECK(squared(1, 1) == 0.0f);

  float sum = squared.Aggregate<float>([](float x) { return x; });
  CHECK(sum == -34.0f);

  int count = squared.Aggregate<int>([](float x) { return x >= 0.0f ? 1 : 0; });
  CHECK(count == 1);
}

TEST_CASE("Matrix column sum") {
  auto a = nnn::FloatMatrix::Create(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
  auto result = nnn::FloatMatrix::SumColumns(a.value());

  CHECK(result(0, 0) == 3);
  CHECK(result(1, 0) == 7);
}
#include <catch2/catch_test_macros.hpp>
#include "main.cuh"

TEST_CASE("Add two ECC points")
{
    ECC_Point p1 = ECC_Point{2, 5};
    ECC_Point p2 = ECC_Point{0, 5};
    ECC_Point result;
    result = pointAdd(p1, p2, P);

    REQUIRE(result.x == 5);
    REQUIRE(result.y == 2);
}

TEST_CASE("Double the point")
{
    ECC_Point p1 = ECC_Point{5197, 3465};
    ECC_Point result;
    result = pointDouble(p1, 3, 7817);

    REQUIRE(result.x == 2539);
    REQUIRE(result.y == 3254);
}

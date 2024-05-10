#include <catch2/catch_test_macros.hpp>
#include "main.cuh"

#define FIELD_ORDER 0xD3915
#define CURVE_A_PARAM 0xb44bc
#define CURVE_B_PARAM 0xa999a

TEST_CASE("Add two points")
{
    // All expected values are based on results in SageMath
    SECTION("Case 1")
    {
        ECC_point p1 = ECC_point{106721, 485050};
        ECC_point p2 = ECC_point{478200, 808318};
        ECC_point result;
        result = add_points(p1, p2, FIELD_ORDER);

        REQUIRE(result.x == 673570);
        REQUIRE(result.y == 379890);
    }
    SECTION("Case 2")
    {
        ECC_point p1 = ECC_point{129876, 188556};
        ECC_point p2 = ECC_point{39538, 547853};
        ECC_point result;
        result = add_points(p1, p2, FIELD_ORDER);

        REQUIRE(result.x == 380300);
        REQUIRE(result.y == 620732);
    }
    SECTION("Case 3")
    {
        ECC_point p1 = ECC_point{195914, 577386};
        ECC_point p2 = ECC_point{589027, 802679};
        ECC_point result;
        result = add_points(p1, p2, FIELD_ORDER);

        REQUIRE(result.x == 412960);
        REQUIRE(result.y == 74681);
    }
}

// TEST_CASE("Double the point")
// {
//     ECC_point p1 = ECC_point{129876, 129876};
//     ECC_point result;
//     result = mul_point(p1,CURVE_A_PARAM, FIELD_ORDER);
//
//     REQUIRE(result.x == 2539);
//     REQUIRE(result.y == 3254);
// }

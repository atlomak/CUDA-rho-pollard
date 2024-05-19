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
        ECC_point p1 = ECC_point{184224, 74658};
        ECC_point p2 = ECC_point{428817, 567437};
        ECC_point result;
        result = add_points(p1, p2, FIELD_ORDER);

        REQUIRE(result.x == 109605);
        REQUIRE(result.y == 690162);
    }
    SECTION("Case 2")
    {
        ECC_point p1 = ECC_point{24069, 233375};
        ECC_point p2 = ECC_point{249867, 503874};
        ECC_point result;
        result = add_points(p1, p2, FIELD_ORDER);

        REQUIRE(result.x == 847840);
        REQUIRE(result.y == 636963);
    }
    SECTION("Case 3")
    {
        ECC_point p1 = ECC_point{40300, 763164};
        ECC_point p2 = ECC_point{18900, 353015};
        ECC_point result;
        result = add_points(p1, p2, FIELD_ORDER);

        REQUIRE(result.x == 548652);
        REQUIRE(result.y == 419566);
    }
}

TEST_CASE("Double the point")
{
    ECC_point p1 = ECC_point{264320, 549393};
    ECC_point result;
    result = mul_point(p1,CURVE_A_PARAM, FIELD_ORDER);

    REQUIRE(result.x == 38956);
    REQUIRE(result.y == 83726);
}

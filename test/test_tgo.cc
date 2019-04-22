//
// Created by lei on 4/22/19.
//

#include "catch.hpp"

#include "tgo.hpp"

#include <armadillo>
#include <vector>
#include <cmath>
#include <functional>
#include <fmt/format.h>

int count = 0;

double func(double x) {
    count += 1;
    return std::pow(x - 0.5, 2);
}

std::function<double(double)> TGO::functor_=func;

TEST_CASE("x^2", "[tgo]") {
    int nsample = 1e2;
    int nk = 4;
    double xmin = -1.0;
    double xmax = 1.0;
    double xtol = 1.0e-7;
    TGO tgo(nsample, nk, xmin, xmax, xtol);
    OptimizeResult result = tgo.optimize();
    fmt::print("count: {:d}\n", count);
    REQUIRE(result.x == Approx(0.5));
}

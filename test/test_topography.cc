//
// Created by lei on 4/21/19.
//

#include "catch.hpp"

#include "topography.hpp"
#include <armadillo>
#include <vector>
#include <fmt/format.h>

arma::vec func1(const arma::vec &x, double c = 0.) {
    return arma::pow((x-c), 2);
}


TEST_CASE("f = x^2", "[topo]") {
    int n = 8;
    arma::vec x = arma::linspace(-1.1, 1, n);
    arma::vec f = func1(x);
    Topography topo(x, f, 3);
    std::vector<double> min_pool = topo.minimize_pool();
    double dx = (x(1) - x(0)) / 2.;
    REQUIRE(min_pool[0] == Approx(0.0).margin(dx));
}

TEST_CASE("f = x^2 * (x-1)^2", "[topo]") {
    int n = 100;
    arma::vec x = arma::linspace(-1.1, 1, n);
    arma::vec f = func1(x) % func1(x, 1.0);
    Topography topo(x, f, 3);
    std::vector<double> min_pool = topo.minimize_pool();
    double dx = (x(1) - x(0)) / 2.;
    REQUIRE(min_pool[0] == Approx(0.0).margin(dx));
    REQUIRE(min_pool[1] == Approx(1.0).margin(dx));
}
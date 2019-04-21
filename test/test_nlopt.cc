//
// Created by lei on 4/21/19.
//

#include "catch.hpp"

#include <vector>
#include <cmath>
#include <fmt/format.h>
#include <nlopt.hpp>

double func(const std::vector<double> &x, std::vector<double> &grad,
        void *f_data) {
    if (!grad.empty()) {
        grad[0] = 2.0 * (x[0] - 1.0);
    }
    return std::pow(x[0] - 1.0, 2);
}


TEST_CASE("slsqp", "[nlopt]")
{
  nlopt::opt opt(nlopt::LD_SLSQP, 1);
  std::vector<double> lb{-1.0}, ub{1.0};
  opt.set_min_objective(func, NULL);
  std::vector<double> x{-0.5};
  double fev = 1.0e20;
  try {
    nlopt::result result = opt.optimize(x, fev);
  }
  catch (std::runtime_error &err) {
    fmt::print("{}\n", err.what());
  }
  REQUIRE(x[0] == Approx(1.0));
  REQUIRE(fev == Approx(0.0));
}
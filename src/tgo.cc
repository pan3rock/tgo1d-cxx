//
// Created by lei on 4/21/19.
//

#include "tgo.hpp"
#include "topography.hpp"

#include <armadillo>
#include <nlopt.hpp>
#include <fmt/format.h>
#include <functional>

using namespace std::placeholders;
typedef double (*vfunc)(const std::vector<double> &x,
                        std::vector<double> &grad, void *data);

OptimizeResult::OptimizeResult(const arma::vec &xl,
                               const arma::vec &fl,
                               double x, double f):
  xl(xl), funl(fl), x(x), f(f)
{ }


TGO::TGO(std::function<double(double)> functor,
         int nsample,
         int nk,
         double xmin,
         double xmax,
         double epsilon):
  nsample_(nsample),
  nk_(nk),
  xmin_(xmin),
  xmax_(xmax),
  epsilon_(epsilon)
{
  std::function<double(double)> functor_ = functor;
}


double TGO::func_nlopt(const std::vector<double> &x,
                       std::vector<double> &grad,
                       void *f_data) {
  if (!grad.empty()) {
    grad[0] = approx_grad(x[0]);
  }
  return functor_(x[0]);
}

double TGO::approx_grad(double x) {
  double grad = 0.;
  return grad;
}


OptimizeResult TGO::optimize() {
  arma::vec x = arma::linspace(xmin_, xmax_, nsample_);
  arma::vec fev;
  for (auto i=0; i<nsample_; ++i) {
    fev(i) = functor_(x(i));
  }
  Topography topo(x, fev, nk_);
  std::vector<double> min_pool = topo.minimize_pool();
  std::vector<double> x_local, f_local;

  nlopt::opt opt(nlopt::LD_SLSQP, 1);
  std::vector<double> lb{xmin_}, ub{xmax_};
  opt.set_min_objective(func_nlopt, nullptr);
  for (auto x0 : min_pool) {
    std::vector<double> xl{x0};
    double fl = 1.0e20;
    try {
      nlopt::result result = opt.optimize(xl, fl);
    } catch (std::runtime_error &err) {
      fmt::print("{}\n", err.what());
    }
    x_local.push_back(xl[0]);
    f_local.push_back(fl);
  }
  int num_minimal = x_local.size();
  arma::vec xl(&x_local[0], num_minimal, false, true);
  arma::vec fl(&f_local[0], num_minimal, false, true);
  int ind_best = arma::index_min(fl);
  double x_best = xl(ind_best);
  double f_best = fl(ind_best);
  OptimizeResult result(xl, fl, x_best, f_best);
  return result;
}

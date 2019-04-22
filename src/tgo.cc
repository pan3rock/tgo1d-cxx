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

OptimizeResult::OptimizeResult(const arma::vec &xl,
                               const arma::vec &fl,
                               double x, double f) :
        xl(xl), funl(fl), x(x), f(f) {}


TGO::TGO(int nsample, int nk,
         double xmin, double xmax, double xtol) :
        nsample_(nsample),
        nk_(nk),
        xmin_(xmin),
        xmax_(xmax),
        xtol_(xtol) { }


double TGO::func_nlopt(const std::vector<double> &x,
                       std::vector<double> &grad,
                       void *f_data) {
    double fev = functor_(x[0]);
    if (!grad.empty()) {
        grad[0] = approx_grad(x[0], fev, epsilon_);
    }
    return fev;
}


double TGO::approx_grad(double x, double f, double epsilon) {
    double fev = functor_(x + epsilon);
    double grad = (fev - f) / epsilon;
    return grad;
}


OptimizeResult TGO::optimize() {
    arma::vec x = arma::linspace(xmin_, xmax_, nsample_);
    arma::vec fev(nsample_);
    for (auto i = 0; i < nsample_; ++i) {
        fev(i) = functor_(x(i));
    }
    Topography topo(x, fev, nk_);
    std::vector<double> min_pool = topo.minimize_pool();
    std::vector<double> x_local, f_local;

    nlopt::opt opt(nlopt::LD_SLSQP, 1);
    std::vector<double> lb{xmin_}, ub{xmax_};
    opt.set_min_objective(func_nlopt, nullptr);
    opt.set_xtol_abs(xtol_);
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

//
// Created by lei on 4/21/19.
//

#ifndef TGO1D_CXX_TGO_HPP
#define TGO1D_CXX_TGO_HPP

#include <vector>
#include <functional>
#include <armadillo>

struct OptimizeResult {
    OptimizeResult(const arma::vec &xl,
                   const arma::vec &fl,
                   double x, double f);

    // copy and move
    arma::vec xl;
    arma::vec funl;
    double x;
    double f;
};

class TGO {
public:
    TGO(int nsample, int nk, double xmin, double xmax, double xtol);

    OptimizeResult optimize();

private:
    static double func_nlopt(const std::vector<double> &x,
                             std::vector<double> &grad,
                             void *f_data);
    static double approx_grad(double x, double f, double epsilon);
    static std::function<double(double)> functor_;
    int nsample_;
    int nk_;
    double xmin_;
    double xmax_;
    double xtol_;
};

static double epsilon_ = 1.0e-9;

#endif //TGO1D_CXX_TGO_HPP

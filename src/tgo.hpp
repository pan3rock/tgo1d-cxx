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
    TGO(std::function<double(double)> functor, int nsample, int nk, 
        double xmin, double xmax, double xtol);

    OptimizeResult optimize();

private:
    static double func_nlopt(const std::vector<double> &x,
                             std::vector<double> &grad,
                             void *f_data);
    static double approx_grad(double x, double f, double epsilon);
    std::function<double(double)> functor_;
    int nsample_;
    int nk_;
    double xmin_;
    double xmax_;
    double xtol_;
    static constexpr double epsilon_ = 1.0e-9;

    static TGO *pThis;
};
// so many `static` due to the requirement of static function in nlopt.

#endif //TGO1D_CXX_TGO_HPP

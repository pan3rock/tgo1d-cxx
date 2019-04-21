//
// Created by lei on 4/21/19.
//

#ifndef TGO1D_CXX_TOPOGRAPHY_HPP
#define TGO1D_CXX_TOPOGRAPHY_HPP

#include <vector>
#include <armadillo>

class Topography
{
public:
    Topography(const arma::vec &x, const arma::vec &fev,
            int nk);
    std::vector<double> minimize_pool() const;
private:
    arma::vec x_;
    arma::imat topo_;
};
#endif //TGO1D_CXX_TOPOGRAPHY_HPP

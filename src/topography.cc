#include <utility>

//
// Created by lei on 4/21/19.
//

#include "topography.hpp"

#include <algorithm>
#include <fmt/format.h>
#include <armadillo>


Topography::Topography(const arma::vec &x,
                       const arma::vec &fev, int nk):
                       x_(x), topo_(nk, x.size()) {
    if (not x.is_sorted()) {
        fmt::print("{}\n", "input vector is not a sorted one.");
        exit(-1);
    }
    int nx = x.size();
    int nk2 = 2*nk;
    for (auto i=0; i<nx; ++i) {
        int ind_beg;
        if (i < nk) {
            ind_beg = 0;
        } else if (i >= nx-nk) {
            ind_beg = nx - 1 - nk2;
        } else {
            ind_beg = i - nk;
        }
        arma::vec x_slice = x.subvec(ind_beg, ind_beg+nk2);
        x_slice = arma::abs(x_slice -x[i]);
        arma::uvec ind_sort = arma::sort_index(x_slice);
        for (auto j=0; j<nk; ++j) {
            int ind_global = ind_sort(j+1) + ind_beg;
            double f_diff = fev(ind_global) - fev(i);
            int sign = (f_diff > 0 )  - (f_diff < 0);
            topo_(j, i) = sign * ind_global;
        }
    }
}


std::vector<double> Topography::minimize_pool() const {
    std::vector<double> min_pool;
    for (auto i=0U; i<x_.size(); ++i) {
        if (arma::all(topo_.col(i) > 0)) {
            min_pool.push_back(x_(i));
        }
    }
    return min_pool;
}

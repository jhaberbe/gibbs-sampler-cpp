#include "distributions.h"
#include <cmath>
#include <numeric>
#include <iostream>
#include <stdexcept>
#include <dlib/optimization.h>

NegativeBinomialDistribution::NegativeBinomialDistribution(double initial_log_mean, double initial_log_dispersion) 
    : log_mean_(initial_log_mean), log_dispersion_(initial_log_dispersion) {}

void NegativeBinomialDistribution::fit(std::vector<double>& data) {
    if (data.empty()) throw std::invalid_argument("Empty data in fit()");

    // Compute initial guesses
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / data.size();
    dlib::matrix<double, 2, 1> starting_point;
    starting_point(0) = std::log(mean); // log(mu)
    starting_point(1) = std::log(1.0);  // log(r)

    // Negative log-likelihood function for dlib
    auto nll = [&](const dlib::matrix<double,2,1>& params) {
        double log_mu = params(0);
        double log_r  = params(1);

        double mu = std::exp(log_mu);
        double r  = std::exp(log_r);

        if (mu <= 0.0 || r <= 0.0 || !std::isfinite(mu) || !std::isfinite(r)) {
            return 1e100;  // Penalize invalid params
        }

        double nll_sum = 0.0;
        for (double k : data) {
            double p = r / (r + mu);

            if (p <= 0.0 || p >= 1.0) return 1e100;  // p must be in (0, 1)

            double log_pmf = (
                std::lgamma(k + r)
                - std::lgamma(k + 1)
                - std::lgamma(r)
                + r * std::log(p)
                + k * std::log(1 - p)
            );

            if (!std::isfinite(log_pmf)) return 1e100;  // Defensive

            nll_sum -= log_pmf;
        }

        if (!std::isfinite(nll_sum)) return 1e100;
        return nll_sum;
    };


    // Call dlib optimizer: find_min_using_approximate_derivatives
    try {
        dlib::find_min_using_approximate_derivatives(
            dlib::bfgs_search_strategy(),
            dlib::objective_delta_stop_strategy(1e-3),
            nll, starting_point, -1
        );

        // On success: update parameters
        log_mean_ = starting_point(0);
        log_dispersion_ = starting_point(1);
    } catch (std::exception& e) {
        throw std::runtime_error(std::string("dlib optimization failed: ") + e.what());
    }
}


double NegativeBinomialDistribution::logpmf(double k) const {
    double mu = std::exp(log_mean_);
    double r = std::exp(log_dispersion_);

    double p = r / (r + mu);

    // Using standard negative binomial log pmf
    double log_pmf = (
        std::lgamma(k + r)
        - std::lgamma(k + 1)
        - std::lgamma(r)
        + r * std::log(p)
        + k * std::log(1 - p)
    );

    return log_pmf;
}

double NegativeBinomialDistribution::log_likelihood(std::vector<double> &data) {
    double log_likelihood = 0.0;
    for (const auto &x : data) {
        log_likelihood += logpmf(x);
    }
    return std::exp(log_likelihood);
}

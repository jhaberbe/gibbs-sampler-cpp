#ifndef IBP_H
#define IBP_H

#include "../io/csv.h"
#include "../distributions/distributions.h"

#include <string>
#include <iostream>
#include <unordered_map>
#include <vector>

// FIRST: IBPFactor
class IBPFactor
{
public:
    IBPFactor(size_t n_rows, double alpha);
    ~IBPFactor() = default;

    std::vector<double> latent() const;
    double row_latent(size_t row_index) const;

    std::vector<double> loading() const;
    double row_loading(size_t row_index) const;

    std::vector<double> eta() const;
    double row_eta(size_t row_index) const;

    void row_latent_update(size_t row_index, double value);
    void latent_update(const std::vector<double>& values);

    void row_loading_update(size_t row_index, double value);
    void loading_update(const std::vector<double>& values);

    int member_count() const;

private:
    double alpha_;
    std::vector<double> latent_;
    std::vector<double> loading_;
};

// THEN: IBP
class IBP
{
public:
    IBP(CSV csv, double alpha);
    ~IBP() = default;

    void add_factor();
    void remove_factor(int factor_id);

    double log_likelihood();
    double row_log_likelihood(size_t row_index) const;

    void sample_new_factors(size_t index);
    void update_latent_membership(size_t index);
    void update_latent_loadings();

    void gibbs_update();
    void run(size_t n_iterations, bool verbose = true);

private:
    CSV csv_;
    double alpha_;

    std::vector<double> log_mean_;
    std::vector<double> log_dispersion_;
    std::vector<double> size_factor_;

    std::unordered_map<int, IBPFactor> factors_;  // this now works because IBPFactor is known
};

#endif // IBP_H
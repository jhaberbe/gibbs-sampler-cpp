#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include <vector>

class Distribution 
{
public:
    virtual ~Distribution() {};

    // Pure virtual functions for the distribution interface
    virtual void fit(std::vector<double>& data) = 0;

    // Pure virtual functions for likelihood calculations
    virtual double logpmf(double x) const = 0;
    virtual double log_likelihood(std::vector<double> &data) = 0;
};

class NegativeBinomialDistribution : public Distribution  
{
public:
    NegativeBinomialDistribution(double initial_log_mean = 0.0, double initial_log_dispersion = 0.0);
    ~NegativeBinomialDistribution() override {};

    // Implement the pure virtual functions from the base class
    void fit(std::vector<double>& data) override;

    double logpmf(double x) const override;
    double log_likelihood(std::vector<double> &data) override;

    double log_mean() const { return log_mean_; }
    double log_dispersion() const { return log_dispersion_; }

private:
    // log(mu)
    double log_mean_;
    // log(r)
    double log_dispersion_;
};

#endif

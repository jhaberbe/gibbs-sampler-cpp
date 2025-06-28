#include "ibp.h"
#include <unordered_map>
#include <vector>

IBPFactor::IBPFactor(size_t nrows, double alpha)
    : latent_(nrows, 0.0), loading_(nrows, 0.0), alpha_(alpha) {};

// Eta
double IBPFactor::row_eta(size_t row_index) const {
    if (row_index < latent_.size()) {
        return latent_[row_index] * loading_[row_index];
    }
    throw std::out_of_range("Row index out of range");
};

std::vector<double> IBPFactor::eta() const {
    std::vector<double> total(latent_.size());
    for (size_t i = 0; i < latent_.size(); ++i) {
        total[i] = latent_[i] * loading_[i];
    }
    return total;
};

// Latent
std::vector<double> IBPFactor::latent() const {
    return latent_;
};

double IBPFactor::row_latent(size_t row_index) const {
    if (row_index < latent_.size()) {
        return latent_[row_index];
    }
    throw std::out_of_range("Row index out of range");
};

// Loadings
std::vector<double> IBPFactor::loading() const {
    return loading_;
};

double IBPFactor::row_loading(size_t row_index) const {
    if (row_index < loading_.size()) {
        return loading_[row_index];
    }
    throw std::out_of_range("Row index out of range");
};

// (S/G)etters for the latent vectors.
void IBPFactor::row_latent_update(size_t row_index, double value) {
    if (row_index < latent_.size()) {
        latent_[row_index] = value;
    } else {
        throw std::out_of_range("Row index out of range");
    }
};

void IBPFactor::latent_update(const std::vector<double>& values) {
    if (values.size() == latent_.size()) {
        latent_ = values;
    } else {
        throw std::invalid_argument("Values size does not match latent size");
    }
};

// (S/G)etters for the loading vectors.
void IBPFactor::row_loading_update(size_t row_index, double value) {
    if (row_index < loading_.size()) {
        loading_[row_index] = value;
    } else {
        throw std::out_of_range("Row index out of range");
    }
};

void IBPFactor::loading_update(const std::vector<double>& values) {
    if (values.size() == loading_.size()) {
        loading_ = values;
    } else {
        throw std::invalid_argument("Values size does not match loading size");
    }
};

int IBPFactor::member_count() const {
    int count = 0;
    for (const auto& value : latent_) {
        if (value > 0.5) {
            ++count;
        }
    }
    return count;
};
#include "ibp.h"
#include "../io/csv.h"
#include "../sampler/polyagamma.h"

#include <vector>
#include <unordered_set>

#include <cmath>
#include <numeric>  // for std::accumulate

IBP::IBP(const CSV csv, double alpha)
    : csv_(csv), alpha_(alpha)
{
    size_t n_rows = csv_.nrows();
    size_t n_cols = csv_.ncols();
    std::cout << "[IBP] Constructing model with " << n_rows << " rows and " << n_cols << " columns.\n";

    log_mean_.resize(n_rows, 0.0);
    log_dispersion_.resize(n_rows, 0.0);
    size_factor_.resize(n_rows, 1.0);

    std::vector<double> row_sums(n_rows, 0.0);

    std::cout << "[IBP] Computing row sums...\n";
    for (size_t i = 0; i < n_rows; ++i) {
        double sum = 0.0;
        const auto& row = csv_.get_row(i);  // fetch row once!
        for (size_t j = 0; j < n_cols; ++j)
            sum += row.at(j);
        row_sums[i] = sum;
    }

    double mean_row_sum = std::accumulate(row_sums.begin(), row_sums.end(), 0.0) / n_rows;

    std::cout << "[IBP] Computing size factors, log means, and log dispersions...\n";
    for (size_t i = 0; i < n_rows; ++i) {
        size_factor_[i] = row_sums[i] / mean_row_sum;
        double mean_i = row_sums[i] / n_cols;
        log_mean_[i] = std::log(std::max(mean_i, 1e-8));

        double var_i = 0.0;

        const auto& row = csv_.get_row(i);  // fetch once!

        for (size_t j = 0; j < n_cols; ++j) {
            double x = row.at(j);
            var_i += (x - mean_i) * (x - mean_i);
        }
        var_i /= (n_cols > 1 ? n_cols - 1 : 1);

        double disp_i = (var_i > mean_i) ? (var_i - mean_i) / (mean_i * mean_i) : 1e-8;
        log_dispersion_[i] = std::log(std::max(disp_i, 1e-8));

        if (i % std::max(n_rows / 10, size_t(1)) == 0) {
            std::cout << "[IBP] Processed row " << i << " / " << n_rows << std::endl;
        }
    }

    std::cout << "[IBP] Initialization complete.\n";
};


void IBP::add_factor() {
    // Collect all occupied keys into a set for fast lookup
    std::unordered_set<int> occupied;
    for (const auto& kv : factors_) {
        occupied.insert(kv.first);
    }

    // Find the smallest non-negative integer not in occupied
    int new_id = 0;
    while (occupied.count(new_id)) {
        ++new_id;
    }

    factors_.emplace(new_id, IBPFactor(csv_.nrows(), alpha_));

}

// Remove a factor by ID
void IBP::remove_factor(int factor_id) {
    auto it = factors_.find(factor_id);
    if (it != factors_.end()) {
        factors_.erase(it);
    }
}

// Compute total log-likelihood
double IBP::log_likelihood() {
    double total = 0.0;
    for (size_t n = 0; n < csv_.nrows(); ++n) {
        total += row_log_likelihood(n);
    }
    return total;
}

// Compute log-likelihood for a single row
double IBP::row_log_likelihood(size_t row_index) const {
    const auto y_n = csv_.get_row(row_index);
    size_t D = y_n.size();
    double ll = 0.0;

    std::vector<double> log_mu(D, size_factor_.at(row_index));
    for (const auto& kv : factors_) {
        const auto& factor = kv.second;
        double z = factor.row_latent(row_index);
        for (size_t d = 0; d < D; ++d) {
            log_mu[d] += z * factor.row_loading(d);
        }
    }

    for (size_t d = 0; d < D; ++d) {
        double mu = std::exp(log_mu[d]);
        double r = std::exp(log_dispersion_[d]);
        double y = y_n[d];
        ll += std::lgamma(y + r) - std::lgamma(r) - std::lgamma(y + 1) +
              r * std::log(r / (r + mu)) +
              y * std::log(mu / (r + mu));
    }
    return ll;
}

// Sample new features for a given row
void IBP::sample_new_factors(size_t index) {
    static std::mt19937 rng(std::random_device{}());
    std::poisson_distribution<int> poisson(alpha_ / std::log(csv_.nrows()));

    int K_new = poisson(rng);
    for (int k = 0; k < K_new; ++k) {
        int new_id = 0;
        while (factors_.find(new_id) != factors_.end()) ++new_id;


        IBPFactor factor(csv_.nrows(), alpha_);
        factor.row_latent_update(index, 1.0); // assign current row

        factors_.emplace(new_id, std::move(factor));

    }
}

// Update membership Z for a single row
void IBP::update_latent_membership(size_t index) {
    const auto y_n = csv_.get_row(index);
    size_t D = y_n.size();
    size_t N = csv_.nrows();

    std::vector<double> current_log_mu(D, size_factor_.at(index));
    for (const auto& kv : factors_) {
        const auto& factor = kv.second;
        if (factor.row_latent(index) > 0.5) {
            for (size_t d = 0; d < D; ++d) {
                current_log_mu[d] += factor.row_loading(d);
            }
        }
    }

    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    for (auto& kv : factors_) {
        int k = kv.first;
        auto& factor = kv.second;

        bool was_active = factor.row_latent(index) > 0.5;

        std::vector<double> log_mu0 = current_log_mu;
        if (was_active) {
            for (size_t d = 0; d < D; ++d) log_mu0[d] -= factor.row_loading(d);
        }

        std::vector<double> log_mu1 = log_mu0;
        for (size_t d = 0; d < D; ++d) log_mu1[d] += factor.row_loading(d);

        double ll0 = 0.0, ll1 = 0.0;
        for (size_t d = 0; d < D; ++d) {
            double mu0 = std::exp(log_mu0[d]), mu1 = std::exp(log_mu1[d]);
            double r = std::exp(log_dispersion_[d]);
            double y = y_n[d];
            ll0 += std::lgamma(y + r) - std::lgamma(r) - std::lgamma(y + 1) +
                   r * std::log(r / (r + mu0)) + y * std::log(mu0 / (r + mu0));
            ll1 += std::lgamma(y + r) - std::lgamma(r) - std::lgamma(y + 1) +
                   r * std::log(r / (r + mu1)) + y * std::log(mu1 / (r + mu1));
        }

        size_t m_k = 0;
        for (size_t i = 0; i < N; ++i) {
            if (i != index && factor.row_latent(i) > 0.5) ++m_k;
        }

        double log_prior_ratio = std::log((m_k + 1e-10) / (N - m_k + 1e-10));
        double logit_p = log_prior_ratio + (ll1 - ll0);
        double p = 1.0 / (1.0 + std::exp(-logit_p));

        double sample = uniform(rng);
        bool new_active = sample < p;

        if (new_active && !was_active) {
            factor.row_latent_update(index, 1.0);
            for (size_t d = 0; d < D; ++d) current_log_mu[d] += factor.row_loading(d);
        } else if (!new_active && was_active) {
            factor.row_latent_update(index, 0.0);
            for (size_t d = 0; d < D; ++d) current_log_mu[d] -= factor.row_loading(d);
        }
    }
}

void IBP::update_latent_loadings() {
    size_t N = csv_.nrows();
    size_t D = csv_.ncols();

    PolyaGammaSampler pg;  // instantiate once for the entire update

    for (auto& kv : factors_) {
        auto& factor = kv.second;

        // Find active rows for this factor
        std::vector<size_t> active_indices;
        for (size_t n = 0; n < N; ++n) {
            if (factor.row_latent(n) > 0.5) active_indices.push_back(n);
        }
        if (active_indices.empty()) continue;

        // Build offset matrix: (N_active x D)
        std::vector<std::vector<double>> offset(active_indices.size(), std::vector<double>(D, 0.0));
        for (size_t i = 0; i < active_indices.size(); ++i) {
            size_t n = active_indices[i];
            // Add size factor
            for (size_t d = 0; d < D; ++d) offset[i][d] = size_factor_[n];

            // Add contributions from other factors
            for (const auto& kv2 : factors_) {
                if (kv2.first == kv.first) continue;  // skip self
                const auto& other = kv2.second;
                if (other.row_latent(n) > 0.5) {
                    for (size_t d = 0; d < D; ++d) {
                        offset[i][d] += other.row_loading(d);
                    }
                }
            }
        }

        // Compute eta = offset + current A_k
        std::vector<std::vector<double>> eta(active_indices.size(), std::vector<double>(D, 0.0));
        for (size_t i = 0; i < active_indices.size(); ++i) {
            for (size_t d = 0; d < D; ++d) {
                eta[i][d] = offset[i][d] + factor.row_loading(d);
            }
        }

        // Construct Y: observed counts (N_active x D)
        std::vector<std::vector<double>> Y(active_indices.size(), std::vector<double>(D, 0.0));
        for (size_t i = 0; i < active_indices.size(); ++i) {
            Y[i] = csv_.get_row(active_indices[i]);
        }

        // Sample omega: Polya-Gamma variables (N_active x D)
        std::vector<std::vector<double>> omega(active_indices.size(), std::vector<double>(D, 1.0));
        for (size_t i = 0; i < active_indices.size(); ++i) {
            for (size_t d = 0; d < D; ++d) {
                double y_nd = Y[i][d];
                double r = std::exp(log_dispersion_[d]);
                double b = y_nd + r;
                double c = eta[i][d];

                // optional numerical stability clamp
                c = std::max(std::min(c, 30.0), -30.0);

                omega[i][d] = pg.sample(b, c, 200);
            }
        }

        // Posterior update for each dimension d
        for (size_t d = 0; d < D; ++d) {
            double XWX = 0.0, Xy = 0.0;
            for (size_t i = 0; i < active_indices.size(); ++i) {
                double omega_val = omega[i][d];
                XWX += omega_val;  // X=1 per active row
                Xy += (Y[i][d] - std::exp(log_dispersion_[d])) / 2.0;
            }

            double precision = XWX + 0.01;  // prior precision
            double posterior_var = 1.0 / precision;
            double posterior_mean = posterior_var * Xy;

            static std::mt19937 rng(std::random_device{}());
            std::normal_distribution<double> normal(posterior_mean, std::sqrt(posterior_var));
            factor.row_loading_update(d, normal(rng));
        }
    }
}


void IBP::gibbs_update(size_t current_iter, size_t remove_every) {
    std::cout << "[IBP] Starting Gibbs update step (iteration " << current_iter << ")...\n";

    size_t total = csv_.nrows();
    for (size_t n = 0; n < total; ++n) {
        update_latent_membership(n);
        sample_new_factors(n);
        if (n % 100 == 1){update_latent_loadings();}

        // More frequent progress bar: update every ~0.1% or at end
        if (n % (std::max(total / 1000, size_t(1))) == 0 || n + 1 == total) {
            double progress = static_cast<double>(n + 1) / total;
            int barWidth = 50;
            std::cout << "\r[";
            int pos = static_cast<int>(barWidth * progress);
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << " %"
                      << " | Factors: " << factors_.size() << std::flush;
        }
    }
    std::cout << "\n[IBP] Finished Gibbs update.\n";

    // Remove empty factors every remove_every iterations
    if (remove_every > 0 && (current_iter + 1) % remove_every == 0) {
        std::cout << "[IBP] Removing empty factors at iteration " << current_iter + 1 << "...\n";
        std::vector<int> to_remove;
        for (const auto& kv : factors_) {
            const auto& factor = kv.second;
            size_t active = factor.member_count();
            if (active <= 1) to_remove.push_back(kv.first);
        }
        for (int k : to_remove) {
            std::cout << "[IBP] Removing empty factor " << k << "\n";
            remove_factor(k);
        }
    }
}


void IBP::run(size_t n_iterations, bool verbose, size_t remove_every) {
    std::cout << "[IBP] Starting Gibbs sampling for " << n_iterations << " iterations.\n";
    for (size_t iter = 0; iter < n_iterations; ++iter) {
        std::cout << "[IBP] ---------------------------\n";
        std::cout << "[IBP] Iteration " << iter << "\n";

        gibbs_update(iter, remove_every);

        if (verbose) {
            std::cout << "[IBP] Iteration " << iter
                      << " complete: Factors=" << factors_.size()
                      << ", Log-likelihood=" << log_likelihood() << "\n";
        }
    }
    std::cout << "[IBP] Gibbs sampling complete.\n";
}

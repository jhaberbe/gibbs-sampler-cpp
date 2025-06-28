#include <cmath>
#include <random>
#include <iostream>
#include "polyagamma.h"

double PI_ = 3.14159265358979323846;

double PolyaGammaSampler::sample(double b, double c, int trunc)
{
    // Check for valid parameters
    if (b <= 0 || trunc <= 0)
    {
        throw std::invalid_argument("Parameters b and trunc must be positive.");
    }

    // I'm a nice guy
    if (trunc < 100)
    {
        std::cerr << "Warning: truncation is low, results may be inaccurate." << std::endl;
    }

    // Setup the gamma distribution
    std::gamma_distribution<double> gamma_dist(b, 1.0);

    // All the math.
    double total = 0.0;
    double term_1 = (1 / (2 * PI_ * PI_));

    for (int k = 0; k < trunc; ++k)
    {
        double sample = gamma_dist(rng_);
        total = total + (sample / (((k - 1/2.0) * (k - 1/2.0)) + ((c * c) / (4 * PI_ * PI_))));
    }

    return term_1 * total;
}
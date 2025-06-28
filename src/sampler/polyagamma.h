#ifndef POLYAGAMMA_H
#define POLYAGAMMA_H

#include <random>
#include <iostream>

class PolyaGammaSampler
{
public:
    // Constructor
    PolyaGammaSampler(int seed = 42069): rng_(seed) {};

    // Sample from the Polya-Gamma distribution
    double sample(double b, double c, int trunc);
private:
    std::mt19937 rng_;
};

#endif
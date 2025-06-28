#include <iostream>
#include "io/csv.h"
#include "ibp/ibp.h"
#include "sampler/polyagamma.h"
#include "distributions/distributions.h"

int main()
{
    try {
        CSV csv("/Users/jameshaberberger/Gitlab/gibbs-sampler-cpp/data/macrophage-counts.csv");
        csv.read();
        std::cout << csv.nrows() << ", " << csv.ncols() << "\n";

        if (csv.nrows() < 2) {
            std::cerr << "CSV must have at least 2 rows, got " << csv.nrows() << std::endl;
            return 1;
        }

        double alpha = 1.0 / std::log(csv.nrows());
        IBP ibp(csv, alpha);
        ibp.run(1000, true);

    } catch (const std::exception& e) {
        std::cerr << "Error during execution: " << e.what() << std::endl;
        return 1;
    }
}
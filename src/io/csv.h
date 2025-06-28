#ifndef CSV_H
#define CSV_H

#include <string>
#include <vector>
#include <utility>

class CSV
{
public:
    explicit CSV(const std::string& filename);

    // Reading the CSV file.
    void read();

    // Attributes.
    size_t nrows() const;
    size_t ncols() const;
    std::pair<size_t, size_t> shape() const;
    const std::vector<std::string>& row_names() const;
    const std::vector<std::string>& column_names() const;

    // Accessing data.
    double index(size_t row, size_t column) const;
    std::vector<double> get_row(size_t index) const;
    std::vector<double> get_col(size_t index) const;
    const std::vector<std::vector<double>>& data() const;

private:
    std::string filename_;
    std::vector<std::string> row_names_;
    std::vector<std::string> column_names_;
    std::vector<std::vector<double>> data_;
};

#endif
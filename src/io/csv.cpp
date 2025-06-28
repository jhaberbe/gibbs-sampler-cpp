#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include "csv.h"

CSV::CSV(const std::string& filename): filename_(filename){};

void CSV::read()
{
    std::ifstream file(filename_);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename_);
    }

    data_.clear();
    column_names_.clear();
    row_names_.clear();

    std::string line;
    size_t line_number = 0;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        if (line_number == 0) {
            while (std::getline(ss, cell, ',')) {
                column_names_.push_back(cell);
            }
        } else {
            std::getline(ss, cell, ',');
            row_names_.push_back(cell);

            while (std::getline(ss, cell, ',')) {
                row.push_back(std::stoi(cell));
            }
            data_.push_back(row);
        }
        ++line_number;
    }

    file.close();
};

size_t CSV::nrows() const {
    return data_.size();
};

size_t CSV::ncols() const {
    if (data_.empty()) {
        return 0; 
    }
    return data_.at(0).size();
};

std::pair<size_t, size_t> CSV::shape() const {
    return {nrows(), ncols()};
};

const std::vector<std::vector<double>>& CSV::data() const {
    return data_;
};

std::vector<double> CSV::get_row(size_t index) const {
    if (index >= nrows()) {
        throw std::out_of_range("Row index out of range");
    }
    return data_.at(index);
};

std::vector<double> CSV::get_col(size_t index) const {
    if (index >= ncols()) {
        throw std::out_of_range("Column index out of range");
    }
    std::vector<double> column;
    for (const auto& row : data_) {
        column.push_back(row.at(index));
    }
    return column;
};

const std::vector<std::string>& CSV::row_names() const
{
    return row_names_;
};

const std::vector<std::string>& CSV::column_names() const
{
    return column_names_;
};

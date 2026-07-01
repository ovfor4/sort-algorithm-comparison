#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

struct Options {
    std::string output_path;
    std::size_t max_n = 1000000;
    std::size_t dense_n_max = 500;
    std::size_t log_base = 2;
    std::size_t log_steps = 4;
    std::size_t k_scale = 128;
    std::size_t k_min = 8;
    std::size_t max_insertion_n = 4096;
};

struct Dataset {
    std::size_t n = 0;
    std::vector<std::string> order;
    std::vector<int> data;
};

constexpr std::uint32_t kSeed = 123456789u;

std::size_t parse_size_arg(const std::string& text, const std::string& name) {
    std::size_t parsed = 0;
    std::size_t consumed = 0;

    try {
        parsed = std::stoull(text, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid value for " + name + ": " + text);
    }

    if (consumed != text.size()) {
        throw std::runtime_error("invalid value for " + name + ": " + text);
    }

    return parsed;
}

void print_usage(const char* program) {
    std::cerr << "usage: " << program
              << " --output <file> [--max-n N] [--dense-n-max N] [--log-base B]"
                 " [--log-steps S] [--k-scale K] [--k-min K] [--max-insertion-n N]\n";
}

Options parse_options(int argc, char* argv[]) {
    Options options;

    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];

        auto require_value = [&](const std::string& option_name) -> std::string {
            if (index + 1 >= argc) {
                throw std::runtime_error("missing value for " + option_name);
            }
            ++index;
            return argv[index];
        };

        if (arg == "--output") {
            options.output_path = require_value(arg);
        } else if (arg == "--max-n") {
            options.max_n = parse_size_arg(require_value(arg), arg);
        } else if (arg == "--dense-n-max") {
            options.dense_n_max = parse_size_arg(require_value(arg), arg);
        } else if (arg == "--log-base") {
            options.log_base = parse_size_arg(require_value(arg), arg);
        } else if (arg == "--log-steps") {
            options.log_steps = parse_size_arg(require_value(arg), arg);
        } else if (arg == "--k-scale") {
            options.k_scale = parse_size_arg(require_value(arg), arg);
        } else if (arg == "--k-min") {
            options.k_min = parse_size_arg(require_value(arg), arg);
        } else if (arg == "--max-insertion-n") {
            options.max_insertion_n = parse_size_arg(require_value(arg), arg);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.output_path.empty()) {
        throw std::runtime_error("--output is required");
    }
    if (options.max_n < 2) {
        throw std::runtime_error("--max-n must be at least 2");
    }
    if (options.max_n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("--max-n is too large for int dataset values");
    }
    if (options.log_base < 2) {
        throw std::runtime_error("--log-base must be at least 2");
    }
    if (options.log_steps == 0) {
        throw std::runtime_error("--log-steps must be positive");
    }
    if (options.k_min == 0) {
        throw std::runtime_error("--k-min must be positive");
    }

    return options;
}

std::vector<std::size_t> make_n_values(const Options& options) {
    std::set<std::size_t> values;
    const std::size_t dense_end = std::min(options.dense_n_max, options.max_n);

    for (std::size_t n = 2; n <= dense_end; ++n) {
        values.insert(n);
    }

    for (std::size_t step = 0;; ++step) {
        const long double exponent = static_cast<long double>(step) / static_cast<long double>(options.log_steps);
        const long double raw_n = std::pow(static_cast<long double>(options.log_base), exponent);

        if (raw_n > static_cast<long double>(options.max_n) + 0.5L) {
            break;
        }

        const auto n = static_cast<std::size_t>(std::llround(raw_n));
        if (n > options.dense_n_max && n >= 2 && n <= options.max_n) {
            values.insert(n);
        }
    }

    return {values.begin(), values.end()};
}

std::size_t rounds_for_n(std::size_t n, const Options& options) {
    const double base = static_cast<double>(options.log_base);
    const double safe_n = static_cast<double>(std::max(n, options.log_base));
    const double log_value = std::log(safe_n) / std::log(base);
    const double scaled = static_cast<double>(options.k_scale) / std::sqrt(log_value);
    const auto rounds = static_cast<std::size_t>(std::ceil(scaled));
    return std::max(options.k_min, rounds);
}

std::vector<std::string> make_algorithm_order(std::size_t n, const Options& options, std::mt19937& rng) {
    std::vector<std::string> order;
    if (n <= options.max_insertion_n) {
        order.push_back("insertion");
    }
    order.push_back("quick");
    order.push_back("combined");

    std::shuffle(order.begin(), order.end(), rng);
    return order;
}

std::vector<Dataset> make_datasets(const Options& options) {
    std::mt19937 rng(kSeed);
    std::vector<Dataset> datasets;

    for (const std::size_t n : make_n_values(options)) {
        const std::size_t rounds = rounds_for_n(n, options);
        for (std::size_t round = 0; round < rounds; ++round) {
            Dataset dataset;
            dataset.n = n;
            dataset.order = make_algorithm_order(n, options, rng);
            dataset.data.resize(n);

            std::iota(dataset.data.begin(), dataset.data.end(), 0);
            std::shuffle(dataset.data.begin(), dataset.data.end(), rng);

            datasets.push_back(std::move(dataset));
        }
    }

    std::shuffle(datasets.begin(), datasets.end(), rng);
    return datasets;
}

void write_datasets(const std::vector<Dataset>& datasets, const std::string& path) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot open output file: " + path);
    }

    for (const Dataset& dataset : datasets) {
        output << dataset.n << '\n';
        output << dataset.order.size();
        for (const std::string& name : dataset.order) {
            output << ' ' << name;
        }
        output << '\n';

        for (std::size_t index = 0; index < dataset.data.size(); ++index) {
            if (index != 0) {
                output << ' ';
            }
            output << dataset.data[index];
        }
        output << '\n';
    }
}

int real_main(int argc, char* argv[]) {
    const Options options = parse_options(argc, argv);
    const std::vector<Dataset> datasets = make_datasets(options);
    write_datasets(datasets, options.output_path);

    std::cerr << "wrote " << datasets.size() << " datasets to " << options.output_path << '\n';
    return 0;
}

int main(int argc, char* argv[]) {
    try {
        return real_main(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }
}

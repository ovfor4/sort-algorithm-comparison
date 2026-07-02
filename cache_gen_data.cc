#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Options {
    std::string output_path;
    std::size_t start_n = 2;
    std::size_t max_n = 1000;
    std::uint64_t target_work = 300000000ULL;
    std::size_t k_min = 1;
    std::size_t k_max = 200;
    std::size_t value_max = 100;
    std::uint32_t seed = 123456789u;
};

constexpr std::array<const char*, 6> kLoopOrders{{"ijk", "ikj", "jik", "jki", "kij", "kji"}};

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

std::uint64_t parse_u64_arg(const std::string& text, const std::string& name) {
    std::uint64_t parsed = 0;
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
              << " --output <file> [--start-n N] [--max-n N] [--target-work W] [--k-min K] [--k-max K]"
                 " [--value-max V] [--seed S]\n";
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
        } else if (arg == "--start-n") {
            options.start_n = parse_size_arg(require_value(arg), arg);
        } else if (arg == "--max-n") {
            options.max_n = parse_size_arg(require_value(arg), arg);
        } else if (arg == "--target-work") {
            options.target_work = parse_u64_arg(require_value(arg), arg);
        } else if (arg == "--k-min") {
            options.k_min = parse_size_arg(require_value(arg), arg);
        } else if (arg == "--k-max") {
            options.k_max = parse_size_arg(require_value(arg), arg);
        } else if (arg == "--value-max") {
            options.value_max = parse_size_arg(require_value(arg), arg);
        } else if (arg == "--seed") {
            const std::size_t seed = parse_size_arg(require_value(arg), arg);
            if (seed > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("--seed is too large for uint32_t");
            }
            options.seed = static_cast<std::uint32_t>(seed);
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
    if (options.start_n < 2) {
        throw std::runtime_error("--start-n must be at least 2");
    }
    if (options.max_n < options.start_n) {
        throw std::runtime_error("--max-n must be at least --start-n");
    }
    if (options.target_work == 0) {
        throw std::runtime_error("--target-work must be positive");
    }
    if (options.k_min == 0) {
        throw std::runtime_error("--k-min must be positive");
    }
    if (options.k_max < options.k_min) {
        throw std::runtime_error("--k-max must be at least --k-min");
    }
    if (options.value_max > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("--value-max is too large for int matrix values");
    }

    return options;
}

std::vector<std::size_t> make_n_values(const Options& options) {
    std::vector<std::size_t> values;
    values.reserve(options.max_n - options.start_n + 1);

    for (std::size_t n = options.start_n; n <= options.max_n; ++n) {
        values.push_back(n);
    }

    return values;
}

std::size_t rounds_for_n(std::size_t n, const Options& options) {
    const long double cube = static_cast<long double>(n) * static_cast<long double>(n) * static_cast<long double>(n);
    const long double per_dataset_work = static_cast<long double>(kLoopOrders.size()) * cube;
    const long double raw_rounds = std::ceil(static_cast<long double>(options.target_work) / per_dataset_work);

    const auto unclamped = static_cast<std::size_t>(std::max(1.0L, raw_rounds));
    return std::clamp(unclamped, options.k_min, options.k_max);
}

std::vector<std::size_t> make_schedule(const Options& options) {
    std::vector<std::size_t> schedule;

    for (const std::size_t n : make_n_values(options)) {
        const std::size_t rounds = rounds_for_n(n, options);
        for (std::size_t round = 0; round < rounds; ++round) {
            schedule.push_back(n);
        }
    }

    return schedule;
}

std::array<const char*, 6> make_loop_order(std::mt19937& rng) {
    std::array<const char*, 6> order = kLoopOrders;
    std::shuffle(order.begin(), order.end(), rng);
    return order;
}

void write_matrix(std::ofstream& output, std::size_t n, std::mt19937& rng, std::uniform_int_distribution<int>& dist) {
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            if (col != 0) {
                output << ' ';
            }
            output << dist(rng);
        }
        output << '\n';
    }
}

void write_datasets(const Options& options) {
    std::ofstream output(options.output_path);
    if (!output) {
        throw std::runtime_error("cannot open output file: " + options.output_path);
    }

    std::mt19937 rng(options.seed);
    std::vector<std::size_t> schedule = make_schedule(options);
    std::shuffle(schedule.begin(), schedule.end(), rng);
    std::uniform_int_distribution<int> value_dist(0, static_cast<int>(options.value_max));

    for (const std::size_t n : schedule) {
        const std::array<const char*, 6> order = make_loop_order(rng);

        output << n << '\n';
        output << order.size();
        for (const char* name : order) {
            output << ' ' << name;
        }
        output << '\n';

        write_matrix(output, n, rng, value_dist);
        write_matrix(output, n, rng, value_dist);
    }

    std::cerr << "wrote " << schedule.size() << " datasets to " << options.output_path << '\n';
}

int real_main(int argc, char* argv[]) {
    const Options options = parse_options(argc, argv);
    write_datasets(options);
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

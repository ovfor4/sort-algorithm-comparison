#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using Matrix = std::vector<int>;
using ResultMatrix = std::vector<std::int64_t>;

struct Kernel {
    std::string_view name;
    void (*run)(std::size_t n, const Matrix& a, const Matrix& b, ResultMatrix& c);
};

struct Dataset {
    std::size_t n = 0;
    std::vector<std::string> order;
    Matrix a;
    Matrix b;
};

struct Stats {
    std::uint64_t runs = 0;
    std::uint64_t total_ns = 0;
    std::uint64_t checksum = 0;
};

struct Options {
    std::string input_path;
    bool verify = false;
    std::size_t verify_max_n = 64;
};

void run_ijk(std::size_t n, const Matrix& a, const Matrix& b, ResultMatrix& c) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                c[i * n + j] += static_cast<std::int64_t>(a[i * n + k]) * b[k * n + j];
            }
        }
    }
}

void run_ikj(std::size_t n, const Matrix& a, const Matrix& b, ResultMatrix& c) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            for (std::size_t j = 0; j < n; ++j) {
                c[i * n + j] += static_cast<std::int64_t>(a[i * n + k]) * b[k * n + j];
            }
        }
    }
}

void run_jik(std::size_t n, const Matrix& a, const Matrix& b, ResultMatrix& c) {
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t k = 0; k < n; ++k) {
                c[i * n + j] += static_cast<std::int64_t>(a[i * n + k]) * b[k * n + j];
            }
        }
    }
}

void run_jki(std::size_t n, const Matrix& a, const Matrix& b, ResultMatrix& c) {
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t k = 0; k < n; ++k) {
            for (std::size_t i = 0; i < n; ++i) {
                c[i * n + j] += static_cast<std::int64_t>(a[i * n + k]) * b[k * n + j];
            }
        }
    }
}

void run_kij(std::size_t n, const Matrix& a, const Matrix& b, ResultMatrix& c) {
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                c[i * n + j] += static_cast<std::int64_t>(a[i * n + k]) * b[k * n + j];
            }
        }
    }
}

void run_kji(std::size_t n, const Matrix& a, const Matrix& b, ResultMatrix& c) {
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                c[i * n + j] += static_cast<std::int64_t>(a[i * n + k]) * b[k * n + j];
            }
        }
    }
}

constexpr std::array<Kernel, 6> kKernels{{
    {"ijk", run_ijk},
    {"ikj", run_ikj},
    {"jik", run_jik},
    {"jki", run_jki},
    {"kij", run_kij},
    {"kji", run_kji},
}};

const Kernel* find_kernel(std::string_view name) {
    for (const Kernel& kernel : kKernels) {
        if (kernel.name == name) {
            return &kernel;
        }
    }

    return nullptr;
}

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

std::size_t matrix_element_count(std::size_t n) {
    if (n != 0 && n > std::numeric_limits<std::size_t>::max() / n) {
        throw std::runtime_error("matrix size overflows size_t");
    }
    return n * n;
}

void validate_order(const std::vector<std::string>& order) {
    if (order.size() != kKernels.size()) {
        throw std::runtime_error("algorithm order block must contain exactly 6 loop orders");
    }

    std::set<std::string> seen;
    for (const std::string& name : order) {
        if (find_kernel(name) == nullptr) {
            throw std::runtime_error("unknown loop order in data file: " + name);
        }
        if (!seen.insert(name).second) {
            throw std::runtime_error("duplicate loop order in data file: " + name);
        }
    }
}

bool read_dataset(std::istream& input, Dataset& dataset) {
    dataset = Dataset{};
    if (!(input >> dataset.n)) {
        if (!input.eof()) {
            throw std::runtime_error("invalid dataset size");
        }
        return false;
    }

    if (dataset.n == 0) {
        throw std::runtime_error("dataset size must be positive");
    }

    std::size_t order_count = 0;
    if (!(input >> order_count) || order_count == 0) {
        throw std::runtime_error("invalid algorithm order block");
    }

    dataset.order.resize(order_count);
    for (std::string& name : dataset.order) {
        if (!(input >> name)) {
            throw std::runtime_error("truncated algorithm order block");
        }
    }
    validate_order(dataset.order);

    const std::size_t elements = matrix_element_count(dataset.n);
    dataset.a.resize(elements);
    dataset.b.resize(elements);

    for (int& value : dataset.a) {
        if (!(input >> value)) {
            throw std::runtime_error("truncated matrix A values");
        }
    }
    for (int& value : dataset.b) {
        if (!(input >> value)) {
            throw std::runtime_error("truncated matrix B values");
        }
    }

    return true;
}

std::uint64_t checksum_result(const ResultMatrix& data) {
    std::uint64_t checksum = 0;
    for (std::size_t index = 0; index < data.size(); index += 2048) {
        checksum += static_cast<std::uint64_t>(data[index]);
    }
    if (!data.empty() && (data.size() - 1) % 2048 != 0) {
        checksum += static_cast<std::uint64_t>(data.back());
    }
    return checksum;
}

void print_usage(const char* program) {
    std::cerr << "usage: " << program << " <data-file> [--verify] [--verify-max-n N]\n";
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

        if (arg == "--verify") {
            options.verify = true;
        } else if (arg == "--verify-max-n") {
            options.verify_max_n = parse_size_arg(require_value(arg), arg);
        } else if (options.input_path.empty()) {
            options.input_path = arg;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.input_path.empty()) {
        throw std::runtime_error("data file is required");
    }

    return options;
}

ResultMatrix make_expected(const Dataset& dataset) {
    ResultMatrix expected(matrix_element_count(dataset.n), 0);
    run_ijk(dataset.n, dataset.a, dataset.b, expected);
    return expected;
}

void process_dataset(const Dataset& dataset, const Options& options,
                     std::map<std::pair<std::size_t, std::string>, Stats>& results,
                     std::map<std::size_t, bool>& seen_sizes) {
    seen_sizes[dataset.n] = true;

    ResultMatrix expected;
    const bool verify_full_result = options.verify && dataset.n <= options.verify_max_n;
    if (verify_full_result) {
        expected = make_expected(dataset);
    }

    bool have_checksum_reference = false;
    std::uint64_t checksum_reference = 0;

    for (const std::string& order_name : dataset.order) {
        const Kernel* kernel = find_kernel(order_name);
        if (kernel == nullptr) {
            throw std::logic_error("validated loop order disappeared");
        }

        ResultMatrix work(matrix_element_count(dataset.n), 0);

        std::atomic_signal_fence(std::memory_order_seq_cst);
        const auto start = std::chrono::steady_clock::now();
        kernel->run(dataset.n, dataset.a, dataset.b, work);
        std::atomic_signal_fence(std::memory_order_seq_cst);
        const auto end = std::chrono::steady_clock::now();

        const std::uint64_t checksum = checksum_result(work);
        if (verify_full_result && work != expected) {
            throw std::runtime_error(order_name + " produced an incorrect result");
        }
        if (options.verify && !verify_full_result) {
            if (!have_checksum_reference) {
                have_checksum_reference = true;
                checksum_reference = checksum;
            } else if (checksum != checksum_reference) {
                throw std::runtime_error(order_name + " produced a mismatched checksum");
            }
        }

        const auto elapsed_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

        Stats& stats = results[{dataset.n, std::string(kernel->name)}];
        ++stats.runs;
        stats.total_ns += elapsed_ns;
        stats.checksum += checksum;
    }
}

int real_main(int argc, char* argv[]) {
    const Options options = parse_options(argc, argv);

    std::ifstream input(options.input_path);
    if (!input) {
        throw std::runtime_error("cannot open input file: " + options.input_path);
    }

    std::map<std::pair<std::size_t, std::string>, Stats> results;
    std::map<std::size_t, bool> seen_sizes;

    Dataset dataset;
    std::size_t dataset_count = 0;
    while (read_dataset(input, dataset)) {
        process_dataset(dataset, options, results, seen_sizes);
        ++dataset_count;
    }

    if (dataset_count == 0) {
        throw std::runtime_error("input file contains no datasets");
    }

    std::cout << "n,order,runs,total_ns,avg_ns,checksum,status\n";
    for (const auto& [n, unused] : seen_sizes) {
        (void)unused;
        for (const Kernel& kernel : kKernels) {
            const auto iter = results.find({n, std::string(kernel.name)});
            if (iter == results.end()) {
                std::cout << n << ',' << kernel.name << ",0,0,0,0,skipped\n";
                continue;
            }

            const Stats& stats = iter->second;
            std::cout << n << ',' << kernel.name << ',' << stats.runs << ',' << stats.total_ns << ','
                      << (stats.total_ns / stats.runs) << ',' << stats.checksum << ",ok\n";
        }
    }

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

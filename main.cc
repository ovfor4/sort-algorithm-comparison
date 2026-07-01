#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace borrowed_std {

constexpr int kThreshold = 16;

template <typename RandomIt>
void insertion_sort(RandomIt first, RandomIt last) {
    std::__insertion_sort(first, last, __gnu_cxx::__ops::__iter_less_iter());
}

template <typename RandomIt>
void hybrid_quick_insertion_loop(RandomIt first, RandomIt last) {
    while (last - first > kThreshold) {
        RandomIt cut =
            std::__unguarded_partition_pivot(first, last, __gnu_cxx::__ops::__iter_less_iter());
        hybrid_quick_insertion_loop(cut, last);
        last = cut;
    }
}

template <typename RandomIt>
void hybrid_quick_insertion_sort(RandomIt first, RandomIt last) {
    if (first == last) {
        return;
    }

    // match libstdc++ std::__sort
    // but deliberately removes the heap fallback
    hybrid_quick_insertion_loop(first, last);
    std::__final_insertion_sort(first, last, __gnu_cxx::__ops::__iter_less_iter());
}

template <typename RandomIt>
RandomIt partition_for_pure_quick_sort(RandomIt first, RandomIt last) {
    auto iter_less_iter = __gnu_cxx::__ops::__iter_less_iter();
    auto iter_less_val = __gnu_cxx::__ops::__iter_comp_val(iter_less_iter);
    auto val_less_iter = __gnu_cxx::__ops::__val_comp_iter(iter_less_iter);

    RandomIt mid = first + (last - first) / 2;
    std::__move_median_to_first(first, first + 1, mid, last - 1, iter_less_iter);

    // libstdc++'s internal partition assumes a final insertion-sort cleanup
    // pure quicksort needs one extra pivot-placement step to be self-contained
    using value_type = typename std::iterator_traits<RandomIt>::value_type;
    value_type pivot = std::move(*first);

    RandomIt left = first + 1;
    RandomIt right = last;

    while (true) {
        while (left != last && iter_less_val(left, pivot)) {
            ++left;
        }

        do {
            --right;
        } while (val_less_iter(pivot, right));

        if (!(left < right)) {
            break;
        }

        std::iter_swap(left, right);
        ++left;
    }

    *first = std::move(*right);
    *right = std::move(pivot);
    return right;
}

template <typename RandomIt>
void pure_quick_sort(RandomIt first, RandomIt last) {
    auto iter_less_iter = __gnu_cxx::__ops::__iter_less_iter();

    while (last - first > 1) {
        if (last - first == 2) {
            if (iter_less_iter(first + 1, first)) {
                std::iter_swap(first, first + 1);
            }
            return;
        }

        RandomIt pivot = partition_for_pure_quick_sort(first, last);

        if (pivot - first < last - (pivot + 1)) {
            pure_quick_sort(first, pivot);
            first = pivot + 1;
        } else {
            pure_quick_sort(pivot + 1, last);
            last = pivot;
        }
    }
}

}  // namespace borrowed_std

struct Sorter {
    std::string_view name;
    void (*run)(std::vector<int>& data);
};

struct Dataset {
    std::size_t n = 0;
    std::vector<std::string> order;
    std::vector<int> data;
};

struct Stats {
    std::uint64_t runs = 0;
    std::uint64_t total_ns = 0;
    std::uint64_t checksum = 0;
};

void run_insertion(std::vector<int>& data) {
    borrowed_std::insertion_sort(data.begin(), data.end());
}

void run_pure_quick(std::vector<int>& data) {
    borrowed_std::pure_quick_sort(data.begin(), data.end());
}

void run_hybrid(std::vector<int>& data) {
    borrowed_std::hybrid_quick_insertion_sort(data.begin(), data.end());
}

constexpr std::array<Sorter, 3> kSorters{{
    {"insertion", run_insertion},
    {"quick", run_pure_quick},
    {"combined", run_hybrid},
}};

const Sorter* find_sorter(std::string_view name) {
    for (const Sorter& sorter : kSorters) {
        if (sorter.name == name) {
            return &sorter;
        }
    }

    return nullptr;
}

std::vector<Dataset> load_datasets(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open input file: " + path);
    }

    std::vector<Dataset> datasets;

    while (true) {
        Dataset dataset;
        if (!(input >> dataset.n)) {
            break;
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
            if (find_sorter(name) == nullptr) {
                throw std::runtime_error("unknown sorter in data file: " + name);
            }
        }

        dataset.data.resize(dataset.n);
        for (int& value : dataset.data) {
            if (!(input >> value)) {
                throw std::runtime_error("truncated dataset values");
            }
        }

        datasets.push_back(std::move(dataset));
    }

    if (datasets.empty()) {
        throw std::runtime_error("input file contains no datasets");
    }

    return datasets;
}

std::uint64_t checksum_result(const std::vector<int>& data) {
    std::uint64_t checksum = 0;
    for (std::size_t index = 0; index < data.size(); index += 2048) {
        checksum += static_cast<std::uint64_t>(data[index]);
    }
    if ((data.size() - 1) % 2048 != 0) {
        checksum += static_cast<std::uint64_t>(data.back());
    }
    return checksum;
}

void print_usage(const char* program) {
    std::cerr << "usage: " << program << " <data-file> [--verify]\n";
}

int real_main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string input_path;
    bool verify = false;

    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--verify") {
            verify = true;
        } else if (input_path.empty()) {
            input_path = arg;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (input_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::vector<Dataset> datasets = load_datasets(input_path);
    std::map<std::pair<std::size_t, std::string>, Stats> results;

    for (const Dataset& dataset : datasets) {
        std::vector<int> expected;
        if (verify) {
            expected = dataset.data;
            std::sort(expected.begin(), expected.end());
        }

        for (const std::string& sorter_name : dataset.order) {
            const Sorter* sorter = find_sorter(sorter_name);
            if (sorter == nullptr) {
                throw std::logic_error("validated sorter disappeared");
            }

            std::vector<int> work = dataset.data;

            std::atomic_signal_fence(std::memory_order_seq_cst);
            const auto start = std::chrono::steady_clock::now();
            sorter->run(work);
            std::atomic_signal_fence(std::memory_order_seq_cst);
            const auto end = std::chrono::steady_clock::now();

            if (verify && work != expected) {
                throw std::runtime_error(std::string(sorter->name) + " produced an incorrect result");
            }

            const auto elapsed_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

            Stats& stats = results[{dataset.n, std::string(sorter->name)}];
            ++stats.runs;
            stats.total_ns += elapsed_ns;
            stats.checksum += checksum_result(work);
        }
    }

    std::cout << "n,sorter,runs,total_ns,avg_ns,checksum,status\n";
    std::map<std::size_t, bool> seen_sizes;
    for (const Dataset& dataset : datasets) {
        seen_sizes[dataset.n] = true;
    }

    for (const auto& [n, unused] : seen_sizes) {
        (void)unused;
        for (const Sorter& sorter : kSorters) {
            const auto iter = results.find({n, std::string(sorter.name)});
            if (iter == results.end()) {
                std::cout << n << ',' << sorter.name << ",0,0,0,0,skipped\n";
                continue;
            }

            const Stats& stats = iter->second;
            std::cout << n << ',' << sorter.name << ',' << stats.runs << ',' << stats.total_ns << ','
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
        return 1;
    }
}

# sort-algorithm-comparison

mkdir build
g++ -std=c++23 -O3 -Wall -Wextra -pedantic gen_data.cc -o build/gen_data
g++ -std=c++23 -O3 -Wall -Wextra -pedantic main.cc -o build/benchmark
g++ -std=c++23 -O3 -DNDEBUG gen_data.cc -o gen_data
g++ -std=c++23 -O3 -DNDEBUG main.cc -o benchmark


build/gen_data --output sort_data.txt.local --max-n 100000 --dense-n-max 128 --log-base 2 --log-steps 8 --k-scale 100 --k-min 20
build/benchmark sort_data.txt.local --verify

output:
n,sorter,runs,total_ns,avg_ns,checksum,status

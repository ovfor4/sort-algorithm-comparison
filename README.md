# sort-algorithm-comparison

```
mkdir build
g++ -std=c++23 -O3 -Wall -Wextra -pedantic gen_data.cc -o build/gen_data
g++ -std=c++23 -O3 -Wall -Wextra -pedantic main.cc -o build/benchmark

build/gen_data --output sort_data.txt.local --max-n 100000 --dense-n-max 256 --log-base 2 --log-steps 8 --k-scale 100 --k-min 50
build/benchmark sort_data.txt.local --verify
```

output:
n,sorter,runs,total_ns,avg_ns,checksum,status


matrix:
```
g++ -std=c++23 -O3 -fno-strict-aliasing -Wall -Wextra -pedantic cache_gen_data.cc -o build/cache_gen_data
g++ -std=c++23 -O3 -fno-strict-aliasing -Wall -Wextra -pedantic cache_main.cc -o build/cache_benchmark

build/cache_gen_data --output cache_data.txt.local --start-n 2 --max-n 256 --target-work 100000000 --k-min 10 --k-max 100
build/cache_benchmark cache_data.txt.local --verify
```
output:
n,order,runs,total_ns,avg_ns,checksum,status

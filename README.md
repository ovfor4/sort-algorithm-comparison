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


n=2000
```n,order,runs,total_ns,avg_ns,checksum,status
2000,ijk,1,20826855189,20826855189,9772193422,ok
2000,jik,1,16856232228,16856232228,9772667048,ok
2000,jik,1,17667021004,17667021004,9772193422,ok
2000,jki,1,51555942206,51555942206,9772193422,ok
2000,kji,1,55521376352,55521376352,9772193422,ok
2000,jki,1,51754546864,51754546864,9761994544,ok
2000,ikj,1,8549164757,8549164757,9772193422,ok
2000,ikj,1,8486644961,8486644961,9761994544,ok
2000,kij,1,9593375623,9593375623,9772667048,ok
2000,kij,1,9601048523,9601048523,9772193422,ok
2000,ijk,1,21284365775,21284365775,9761994544,ok
2000,kij,1,9889601095,9889601095,9761994544,ok
2000,jki,1,50957528926,50957528926,9772667048,ok
2000,ikj,1,8524319859,8524319859,9772667048,ok
2000,ijk,1,20838993052,20838993052,9772667048,ok
2000,kji,1,55600193747,55600193747,9761994544,ok
2000,jik,1,17671874270,17671874270,9761994544,ok
2000,kji,1,53996579716,53996579716,9772667048,ok
```

| Order | Performance (Row-major) | Similar Order | Total Time n=2000 | Relative Time |
|---|---|---|---:|---:|
| **ikj** | Best ⚡️ | kij | **8.52 s** | **1.00×** |
| **kij** | Good 👍 | ikj | **9.70 s** | **1.14×** |
| **jik** | Medium 👌 | ijk | **17.40 s** | **2.04×** |
| **ijk** | Medium 👌 | jik | **20.98 s** | **2.46×** |
| **jki** | Slow 🐢 | kji | **51.42 s** | **6.04×** |
| **kji** | Worst 🤮 | jki | **55.04 s** | **6.46×** |

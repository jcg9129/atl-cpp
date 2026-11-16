// Pure-I/O micro-benchmarks (no data structure), to separate parse/format cost
// from the prefix-sum compute in bench.cpp. Backend from backend.hpp.
//
//   -DBENCH_READ read N, then sum N values, print the sum (pure input)
//   -DBENCH_WRITE read N, then generate and print N values (pure output)
#include "backend.hpp"

int main() {
	io_setup();

#if defined(BENCH_READ)
	u32 n = rd();
	u64 s = 0;
	for (u32 i = 0; i < n; ++i) s += rd();
	wr64(s); // checksum: identical across backends iff parsing agrees

#elif defined(BENCH_WRITE)
	u32 n = rd();
	u64 x = 88172645463325252ull; // xorshift, so every backend writes the same stream
	for (u32 i = 0; i < n; ++i) {
		x ^= x << 13;
		x ^= x >> 7;
		x ^= x << 17;
		wr(u32(x % 1'000'000'000u));
	}
#endif

	io_flush();
	return 0;
}

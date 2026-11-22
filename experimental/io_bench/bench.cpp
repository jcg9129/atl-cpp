// Prefix-sum queries keep the compute identical across I/O backends.
// Input uses half-open ranges.
#include <cstddef>
#include <vector>
#include "backend.hpp"

int main() {
	io_setup();
	u32 const n = rd(), q = rd();
	std::vector<u64> prefix(n);
	u64 sum = 0;
	for (auto &value : prefix) value = (sum += rd());
	for (u32 i = 0; i < q; ++i) {
		u32 const l = rd(), r = rd();
		if (l >= r || r > n) return 1;
		wr64(prefix[r - 1] - (l == 0 ? 0 : prefix[l - 1]));
	}
	io_flush();
}

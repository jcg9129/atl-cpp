#include <atl/ds/dynamic_segment_tree.hpp>
#include <functional>
#include <iostream>

int main() {
	atl::dynamic_segment_tree library(1'000'000'000, 0LL, std::plus<>{});
	library.update(0, [](long long x) { return x + 4; });
	library.set(999'999'999, 7);
	std::cout << library.fold() << ' ' << library.fold(1, 999'999'999) << '\n';
}

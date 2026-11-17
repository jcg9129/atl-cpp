#include <atl/ds/segment_tree_beats.hpp>
#include <array>
#include <iostream>

int main() {
	atl::beats_segment_tree<int, long long> library(std::array{4, 2, 7, 1, 9});
	library.chmin(0, 5, 5);
	library.chmax(1, 4, 3);
	library.add(0, 2, 10);
	library.assign(4, 5, 0);
	std::cout << library.sum(0, 5) << ' ' << library.min(0, 5) << ' ' << library.max(0, 5) << '\n';
}

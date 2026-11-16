#include <atl/ds/coordinate_compression.hpp>
#include <array>
#include <iostream>

int main() {
	auto const result = atl::compress(std::array{30, 10, 20, 10});
	for (auto rank : result.ranks) std::cout << rank << '\n';
	std::cout << result.universe.rank(15) << ' ' << result.universe.rank_upper(20) << '\n';
}

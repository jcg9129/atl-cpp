#include <atl/ds/hierarchical_bitset.hpp>
#include <array>
#include <cstddef>
#include <iostream>

int main() {
	atl::hierarchical_bitset set(100, std::array{3, 17, 64});
	std::cout << set.next(4) << ' ' << set.prev(63) << '\n';
	for (std::size_t x = set.min(); x != atl::hierarchical_bitset::npos; x = set.next(x + 1))
		std::cout << x << '\n';
}

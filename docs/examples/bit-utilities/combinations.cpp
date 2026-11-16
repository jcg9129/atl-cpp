#include <atl/bits/bitlow.hpp>
#include <atl/bits/next_bits_combination.hpp>
#include <atl/types/alias.hpp>
#include <iostream>

int main() {
	using atl::uint32;
	static_assert(atl::lowest_one(uint32(40)) == 8);
	std::cout << "lowest: " << atl::lowest_one(uint32(40)) << '\n';
	std::cout << "pairs:";
	for (uint32 mask = 3; mask != 0 && mask < (uint32(1) << 4); mask = atl::next_bits_combination(mask))
		std::cout << ' ' << mask;
	std::cout << '\n';
}

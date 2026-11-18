#include <atl/nt/floor_sum.hpp>
#include <iostream>

int main() {
	std::cout << static_cast<long long>(atl::floor_sum(5, 3, 2, 1)) << ' '
			  << static_cast<long long>(atl::floor_sum(4, 3, -2, -1)) << '\n';
	std::cout << *atl::first_residue_in_range(31, 6, 0, 11, 19) << '\n';
	auto const add = [](unsigned left, unsigned right) { return left + right; };
	std::cout << atl::floor_staircase_product(5, 3, 2, 1, 1u, 10u, 0u, add) << '\n';
}

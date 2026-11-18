#include <atl/nt/multiplicative.hpp>
#include <iostream>
#include <vector>

int main() {
	std::cout << atl::euler_totient_range(13)[12] << ' '
			  << atl::divisor_count_range(13)[12] << ' '
			  << atl::divisor_sum_range(13)[12] << ' '
			  << atl::divisor_power_sum_range<atl::uint64>(13, 2)[12] << '\n';
	auto const squarefree = atl::multiplicative_fn_range<unsigned>(
		13, [](atl::uint64, atl::uint64 exponent, atl::uint64, std::vector<unsigned> const &) {
			return unsigned(exponent == 1);
		});
	std::cout << squarefree[6] << ' ' << squarefree[12] << '\n';
}

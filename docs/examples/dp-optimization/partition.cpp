#include <atl/dp/optimal_partition.hpp>
#include <atl/types/alias.hpp>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
	// split the array into exactly k pieces, a piece costing the square of its sum
	std::vector<atl::int64> const a{4, 1, 3, 2, 6, 1, 1, 5};
	std::vector<atl::int64> prefix{0};
	for (atl::int64 const value : a) prefix.push_back(prefix.back() + value);
	auto const cost = [&prefix](atl::uint32 j, atl::uint32 i) { return (prefix[i] - prefix[j]) * (prefix[i] - prefix[j]); };
	auto const curve = atl::optimal_partition(a.size(), 4, cost); // k = 1 .. 4
	for (std::size_t k = 1; k <= curve.size(); ++k) std::cout << "k = " << k << ": " << curve[k - 1] << '\n';
	auto const three = atl::optimal_partition_cuts(a.size(), 3, cost);
	std::cout << "k = 3 cuts:";
	for (atl::uint32 const b : three.boundaries) std::cout << ' ' << b;
	std::cout << '\n';
}

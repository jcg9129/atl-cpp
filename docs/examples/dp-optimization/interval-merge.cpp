#include <atl/dp/interval_merge.hpp>
#include <atl/types/alias.hpp>
#include <iostream>
#include <vector>

int main() {
	// stone merging: merging the stones [l, r) costs their total weight
	std::vector<atl::int64> const stones{3, 1, 4, 1, 5};
	std::vector<atl::int64> prefix{0};
	for (atl::int64 const weight : stones) prefix.push_back(prefix.back() + weight);
	auto const table = atl::optimal_interval_merge(stones.size(), [&prefix](atl::uint32 l, atl::uint32 r) {
		return prefix[r] - prefix[l];
	});
	std::cout << "total " << table.value(0, stones.size()) << '\n';
	for (auto const &step : atl::optimal_merge_order(table))
		std::cout << '[' << step.left << ", " << step.split << ") + [" << step.split << ", " << step.right << ")\n";
}

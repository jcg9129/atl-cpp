#include <atl/dp/exact_count_optimum.hpp>
#include <atl/dp/optimal_partition.hpp>
#include <atl/dp/segmentation.hpp>
#include <atl/types/alias.hpp>
#include <compare>
#include <iostream>
#include <utility>
#include <vector>

// a segment cost with its segment count, ordered so that ties go to fewer segments
struct counted {
	atl::int64 value = 0, count = 0;

	friend constexpr counted operator+(counted const &a, counted const &b) {
		return {a.value + b.value, a.count + b.count};
	}

	friend constexpr auto operator<=>(counted const &, counted const &) = default;
};

int main() {
	// cut the array into exactly k pieces, a piece costing the square of its sum
	std::vector<atl::int64> const a{4, 1, 3, 2, 6, 1, 1, 5};
	std::vector<atl::int64> prefix{0};
	for (atl::int64 const value : a) prefix.push_back(prefix.back() + value);
	auto const cost = [&prefix](atl::uint32 j, atl::uint32 i) { return (prefix[i] - prefix[j]) * (prefix[i] - prefix[j]); };
	// the count left free, every piece priced: the decision queue on counted values
	auto const solve = [&](atl::int64 price) {
		auto const r = atl::optimal_segmentation(a.size(), [&](atl::uint32 j, atl::uint32 i) { return counted{cost(j, i) + price, 1}; });
		return std::pair{r.value[a.size()].value, r.value[a.size()].count};
	};
	atl::int64 const total = prefix.back();
	for (atl::int64 k = 1; k <= 4; ++k) {
		auto const r = atl::exact_count_optimum(k, atl::int64(-1), total * total + 1, solve);
		std::cout << "k = " << k << ": " << r.value << " (price " << r.price << ")\n";
	}
	std::cout << "layered solver k = 3: " << atl::optimal_partition(a.size(), 3, cost)[2] << '\n';
}

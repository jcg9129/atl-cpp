#include <atl/dp/convex_piecewise_linear_forest.hpp>
#include <atl/types/alias.hpp>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
	using atl::int64;
	using atl::uint32;
	// a rooted tree with parents before children; edge v runs from parent[v] to v
	std::vector<uint32> const parent{0, 0, 0, 1, 1};
	std::vector<int64> const cost{0, 5, 15, 3, 5}, weight{0, 4, 15, 2, 1};
	std::size_t const n = parent.size();
	std::vector<bool> is_leaf(n, true);
	for (std::size_t v = 1; v < n; ++v) is_leaf[parent[v]] = false;

	// bottom-up: f_v(x) is the least cost of making every path below v total x
	atl::convex_piecewise_linear_forest<int64> forest(n);
	std::vector<atl::slope_clip_interval<int64>> range(n);
	for (std::size_t v = n; v-- > 1;) {
		auto const f = forest[forest.id(v)];
		if (is_leaf[v]) {
			f.add_absolute(cost[v], weight[v]); // the indicator of 0 through the edge
			range[v] = {0, 0};
		} else {
			range[v] = f.clip_slopes(-weight[v], weight[v]); // the edge: + weight |t - cost|
			f.translate(cost[v]);
		}
		forest.merge(forest.id(parent[v]), f.id());
	}

	// top-down: the root at its smallest minimizer, each child at its best total
	std::vector<int64> total(n, 0), new_cost(n, 0);
	total[0] = *forest[forest.id(0)].minimizers().lower;
	int64 change = 0;
	for (std::size_t v = 1; v < n; ++v) {
		total[v] = range[v].clamp(total[parent[v]] - cost[v]);
		new_cost[v] = total[parent[v]] - total[v];
		change += weight[v] * (new_cost[v] < cost[v] ? cost[v] - new_cost[v] : new_cost[v] - cost[v]);
	}
	std::cout << "least weighted change: " << change << '\n';
	std::cout << "path total: " << total[0] << '\n';
	std::cout << "new costs:";
	for (std::size_t v = 1; v < n; ++v) std::cout << ' ' << new_cost[v];
	std::cout << '\n';
}

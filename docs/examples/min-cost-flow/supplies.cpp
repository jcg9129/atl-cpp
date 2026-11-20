#include <atl/graph/min_cost_flow.hpp>
#include <iostream>
#include <vector>

int main() {
	using namespace atl::graph;
	std::vector<bounded_flow_arc<int, long long>> const edges{
		{0, 2, 0, 3, 1}, {0, 1, 2, 5, 3}, {1, 2, 0, 5, 3}};
	auto const r = min_cost_feasible_flow(3, edges, std::vector<int>{3, 0, -3});
	if (!r) return 1;
	std::cout << r->value << ' ' << r->cost << '\n';
	for (auto const flow : r->edge_flow) std::cout << flow << '\n';
	std::cout << min_cost_feasible_flow(3, edges, std::vector<int>{1, 0, 0}).has_value() << '\n';
}

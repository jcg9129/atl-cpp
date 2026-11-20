#include <atl/graph/max_flow.hpp>
#include <array>
#include <iostream>
#include <vector>

int main() {
	std::array<atl::graph::bounded_flow_arc<int>, 3> const cycle{{{0, 1, 1, 4}, {1, 2, 0, 2}, {2, 0, 0, 5}}};
	auto const circulation = atl::graph::feasible_flow(3, cycle, std::vector<int>(3, 0));
	std::cout << "circulation: " << circulation.has_value() << '\n';
	std::array<atl::graph::bounded_flow_arc<int>, 2> const path{{{0, 1, 1, 4}, {1, 2, 0, 3}}};
	auto const greatest = atl::graph::bounded_max_flow(3, path, 0, 2);
	auto const least = atl::graph::bounded_min_flow(3, path, 0, 2);
	if (!greatest || !least) return 1;
	std::cout << "range: " << least->value << ' ' << greatest->value << '\n';
	std::array<atl::graph::bounded_flow_arc<int>, 1> const backward{{{1, 0, 3, 5}}};
	std::cout << "backward maximum: " << atl::graph::bounded_max_flow(2, backward, 0, 1)->value << '\n';
}

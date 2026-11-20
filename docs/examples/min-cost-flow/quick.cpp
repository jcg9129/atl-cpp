#include <atl/graph/min_cost_flow.hpp>
#include <iostream>
#include <vector>

int main() {
	using namespace atl::graph;
	std::vector<flow_arc<int, long long>> const edges{{0, 1, 2, 1}, {0, 1, 3, 5}};
	auto const r = min_cost_max_flow(2, edges, 0, 1);
	auto const potential = min_cost_flow_potentials(2, edges, r.edge_flow);
	std::cout << r.value << ' ' << r.cost << '\n';
	std::cout << r.edge_flow[0] << ' ' << r.edge_flow[1] << '\n';
	std::cout << potential.size() << '\n';
}

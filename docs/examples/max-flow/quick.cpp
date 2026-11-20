#include <atl/graph/max_flow.hpp>
#include <array>
#include <cstddef>
#include <iostream>

int main() {
	std::array<atl::graph::flow_arc<int>, 5> const edges{{{0, 1, 3}, {0, 2, 2}, {1, 2, 1}, {1, 3, 2}, {2, 3, 3}}};
	auto const result = atl::graph::max_flow(4, edges, 0, 3);
	std::cout << "value: " << result.value << '\n';
	std::cout << "edge flows:";
	for (int flow : result.edge_flow) std::cout << ' ' << flow;
	std::cout << "\nsource set:";
	for (std::size_t v = 0; v < result.source_side.size(); ++v)
		if (result.source_side[v]) std::cout << ' ' << v;
	std::cout << '\n';
}

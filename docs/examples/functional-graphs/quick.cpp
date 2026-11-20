#include <atl/graph/functional_graph.hpp>
#include <array>
#include <functional>
#include <iostream>

int main() {
	std::array const successors{1u, 2u, 1u, 0u};
	std::array const values{2u, 3u, 5u, 7u};
	atl::graph::functional_graph_successor_index const endpoints(successors);
	atl::graph::functional_graph_walk_index<unsigned, std::plus<unsigned>> const walks(successors, values, 0, {});
	auto const result = walks.walk(3, 4);
	std::cout << endpoints.successor(3, 4) << ' ' << result.fold << '\n';
	auto const &metadata = walks.decomposition();
	std::cout << metadata.cycle_root[3] << ' ' << metadata.depth_to_cycle[3] << '\n';
	auto const zero = walks.walk(3, 0);
	std::cout << zero.vertex << ' ' << zero.fold << '\n';
}

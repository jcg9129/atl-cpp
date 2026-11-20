#include <atl/graph/csr.hpp>
#include <atl/graph/partition.hpp>
#include <atl/graph/traversal.hpp>
#include <array>
#include <iostream>

int main() {
	using namespace atl::graph;
	std::array<arc<>, 3> const arcs{{{0, 1}, {0, 2}, {1, 3}}};
	csr const graph(4, arcs);
	char const *separator = "";
	for (auto const &node : bfs(graph, 0)) {
		std::cout << separator << node.vertex << ':' << node.depth;
		separator = " ";
	}
	std::cout << '\n';
	vertex_partition const partition{{0, 1, 0, 1}, 2};
	auto const groups = partition.grouped();
	separator = "";
	for (auto const v : groups[0]) {
		std::cout << separator << v;
		separator = " ";
	}
	std::cout << '\n';
}

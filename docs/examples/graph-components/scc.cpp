#include <atl/graph/csr.hpp>
#include <atl/graph/scc.hpp>
#include <iostream>
#include <vector>

int main() {
	std::vector<atl::graph::arc<>> const arcs{{0, 1}, {1, 0}, {1, 2}, {2, 3}, {3, 2}};
	atl::graph::csr const graph(4, arcs);
	auto const components = atl::graph::scc(graph);
	std::cout << components.count() << '\n';
	for (auto const &group : components.groups()) {
		bool is_first = true;
		for (auto const vertex : group) {
			if (!is_first) std::cout << ' ';
			std::cout << vertex;
			is_first = false;
		}
		std::cout << '\n';
	}
}

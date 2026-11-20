#include <atl/graph/csr.hpp>
#include <atl/graph/two_edge_connected_components.hpp>
#include <iostream>
#include <vector>

int main() {
	std::vector<atl::graph::arc<>> const arcs{{0, 1}, {0, 1}, {1, 2}};
	auto const graph = atl::graph::undirected_csr(3, arcs);
	auto const components = atl::graph::two_edge_connected_components(graph);
	std::cout << components.count() << '\n';
	for (auto const &arc : arcs)
		if (!components.same(arc.from, arc.to)) std::cout << arc.from << ' ' << arc.to << '\n';
}

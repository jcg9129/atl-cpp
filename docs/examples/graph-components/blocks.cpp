#include <atl/graph/biconnected_components.hpp>
#include <atl/graph/csr.hpp>
#include <iostream>
#include <vector>

int main() {
	std::vector<atl::graph::arc<>> const arcs{{0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 4}, {4, 2}};
	auto const blocks = atl::graph::biconnected_components(atl::graph::undirected_csr(5, arcs));
	std::cout << blocks.count() << '\n';
	auto const cut = blocks.cut_vertex_mask();
	for (atl::graph::vertex_id v = 0; v < cut.size(); ++v)
		if (cut[v]) std::cout << "cut " << v << '\n';
	std::cout << (blocks.edge_block(0, 1) == blocks.edge_block(1, 2)) << '\n';
	std::cout << (blocks.edge_block(0, 1) == blocks.edge_block(2, 3)) << '\n';
}

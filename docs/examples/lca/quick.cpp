#include <atl/graph/lca.hpp>
#include <array>
#include <iostream>

int main() {
	atl::graph::lca const tree(std::array<atl::graph::vertex_id, 6>{atl::graph::none, 0, 0, 1, 1, 2});
	std::cout << tree(3, 4) << ' ' << tree.distance(3, 5) << '\n';
	std::cout << tree.ancestor(3, 2) << ' ' << tree.path_vertex(3, 5, 2) << '\n';
}

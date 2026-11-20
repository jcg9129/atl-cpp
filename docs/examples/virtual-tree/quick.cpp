#include <atl/graph/lca.hpp>
#include <atl/graph/virtual_tree.hpp>
#include <array>
#include <iostream>

int main() {
	atl::graph::lca const tree(std::array<atl::graph::vertex_id, 6>{atl::graph::none, 0, 0, 1, 1, 2});
	auto const result = atl::graph::virtual_tree_of(tree, std::array<atl::graph::vertex_id, 3>{4, 5, 3});
	for (std::size_t i = 0; i < result.vertices.size(); ++i) {
		if (i != 0) std::cout << ' ';
		std::cout << result.vertices[i];
	}
	std::cout << '\n';
	for (std::size_t i = 0; i < result.parent.size(); ++i) {
		if (i != 0) std::cout << ' ';
		if (result.parent[i] == atl::graph::none)
			std::cout << "root";
		else
			std::cout << result.parent[i];
	}
	std::cout << '\n';
}

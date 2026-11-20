#include <atl/graph/bipartite_matching.hpp>
#include <array>
#include <iostream>

int main() {
	std::array<atl::graph::arc<>, 3> const edges{{{0, 0}, {1, 0}, {1, 1}}};
	auto const matching = atl::graph::bipartite_matching(2, 2, edges);
	auto const cover = atl::graph::minimum_vertex_cover(2, 2, edges, matching);
	auto const independent = atl::graph::maximum_independent_set(2, 2, edges, matching);
	auto const roles = atl::graph::matching_edge_roles(2, 2, edges, matching);
	std::cout << "matching " << matching.size << "\ncover " << cover.size
			  << "\nindependent " << independent.size << '\n';
	std::cout << "roles";
	for (auto role : roles) {
		char const *name = role == atl::graph::matching_edge::mandatory ? "mandatory"
						 : role == atl::graph::matching_edge::allowed	? "allowed"
																		: "forbidden";
		std::cout << ' ' << name;
	}
	std::cout << '\n';
}

#include <atl/graph/bipartite_matching.hpp>
#include <array>
#include <iostream>

int main() {
	std::array<atl::graph::arc<>, 4> const edges{{{0, 1}, {0, 2}, {1, 3}, {2, 3}}};
	auto const paths = atl::graph::minimum_path_cover(4, edges);
	auto const chains = atl::graph::minimum_chain_cover(4, edges);
	std::cout << "paths " << paths.paths().size() << "\nchains " << chains.chains().size() << "\nantichain";
	for (atl::uint32 v = 0; v < 4; ++v)
		if (chains.antichain[v]) std::cout << ' ' << v;
	std::cout << '\n';
}

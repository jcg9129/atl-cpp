#include <atl/graph/centroid_bisect.hpp>
#include <atl/graph/centroid_decomposition.hpp>
#include <atl/graph/ternarization.hpp>
#include <iostream>
#include <span>
#include <vector>

int main() {
	using namespace atl::graph;
	std::vector<std::vector<vertex_id>> const tree{{1}, {0, 2}, {1, 3}, {2, 4}, {3}};
	auto const parent = centroid_decomposition(tree);
	std::size_t roots = 0, pairs = 0;
	for (vertex_id p : parent) roots += p == none;
	centroid_bisect(tree, [&](vertex_id, std::span<visit const> a, std::span<visit const> b) {
		pairs += a.size() * b.size();
	});
	auto const bounded = ternarization_of(tree, 0);
	std::cout << roots << ' ' << pairs << ' ' << bounded.real_count << '\n';
}

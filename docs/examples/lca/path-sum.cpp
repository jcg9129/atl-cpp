#include <atl/ds/segment_tree.hpp>
#include <atl/graph/lca.hpp>
#include <array>
#include <functional>
#include <iostream>
#include <vector>

int main() {
	using atl::graph::vertex_id;
	atl::graph::lca const tree(std::array<vertex_id, 6>{atl::graph::none, 0, 0, 1, 1, 2});
	std::vector<int> ordered(tree.size());
	for (vertex_id v = 0; v < tree.size(); ++v) ordered[tree.entry_time(v)] = int(v + 1);
	atl::segment_tree<int, std::plus<>> sums(ordered, 0);
	int answer = 0;
	tree.path(3, 5, [&](atl::uint32 l, atl::uint32 r, bool) { answer += sums.fold(l, r); });
	std::cout << answer << '\n';
	std::cout << sums.fold(tree.entry_time(1), tree.exit_time(1)) << '\n';
}

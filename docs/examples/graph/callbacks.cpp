#include <atl/graph/traversal.hpp>
#include <iostream>
#include <vector>

int main() {
	using namespace atl::graph;
	std::vector<std::vector<vertex_id>> const graph{{1, 2}, {3}, {}, {}};
	char const *separator = "";
	dfs_visit(graph, 0, events([&](vertex_id v, vertex_id) {
				  std::cout << separator << v;
				  separator = " ";
			  }));
	std::cout << '\n';
	int const size = dfs_fold(
		graph, 0, [](visit) { return 1; },
		[](int &acc, int child, visit, visit, vertex_id) { acc += child; });
	std::cout << size << '\n';
	epoch_marker marks(4);
	for (int run = 0; run < 2; ++run) {
		marks.next_run();
		separator = "";
		bfs_visit(graph, 0, marks, [&](vertex_id v, vertex_id, atl::uint32) {
			std::cout << separator << v;
			separator = " ";
		});
		std::cout << '\n';
	}
}

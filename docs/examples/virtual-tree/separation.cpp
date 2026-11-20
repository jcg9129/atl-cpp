#include <atl/graph/lca.hpp>
#include <atl/graph/virtual_tree.hpp>
#include <array>
#include <iostream>
#include <span>
#include <vector>

int main() {
	using atl::graph::vertex_id;
	atl::graph::lca const tree(std::array<vertex_id, 6>{atl::graph::none, 0, 0, 1, 1, 2});
	std::vector<atl::uint8> is_key(tree.size());
	atl::graph::virtual_tree result;
	auto const separate = [&](std::span<vertex_id const> keys) {
		for (vertex_id v : keys) is_key[v] = 1;
		bool is_impossible = false;
		for (vertex_id v : keys) is_impossible |= tree.parent(v) != atl::graph::none && is_key[tree.parent(v)];
		long long answer = -1;
		if (!is_impossible) {
			atl::graph::virtual_tree_of(tree, keys, result);
			std::vector<atl::uint32> open(result.vertices.size());
			answer = 0;
			for (std::size_t i = result.vertices.size(); i-- > 0;) {
				if (is_key[result.vertices[i]]) {
					answer += open[i];
					open[i] = 1;
				} else if (open[i] > 1) {
					++answer;
					open[i] = 0;
				}
				if (result.parent[i] != atl::graph::none) open[result.parent[i]] += open[i] != 0;
			}
		}
		for (vertex_id v : keys) is_key[v] = 0;
		return answer;
	};
	std::cout << separate(std::array<vertex_id, 3>{3, 4, 5}) << '\n';
	std::cout << separate(std::array<vertex_id, 2>{1, 3}) << '\n';
	std::cout << separate(std::array<vertex_id, 3>{0, 3, 5}) << '\n';
}

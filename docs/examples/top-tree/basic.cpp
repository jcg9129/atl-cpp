#include <atl/ds/top_tree_core.hpp>
#include <iostream>
#include <vector>

int main() {
	atl::extremes_top_tree<long long> t(std::vector<long long>{5, 1, 4, 2, 3});
	t.assign_parents(std::vector<atl::uint32>{atl::graph::none, 0, 0, 1, 1});
	t.path_add(3, 2, 10);
	t.subtree_assign(1, 7);
	long long lo = t.subtree_min(0);
	t.make_root(4);
	long long hi = t.subtree_max(0); // the side of 0 under root 4: {0, 2}: 15
	long long s = t.path_sum(2, 3);

	atl::diameter_top_tree<long long> d(4);
	atl::uint32 e = d.link(0, 1, 5);
	d.link(1, 2, 3);
	d.link(1, 3, 4);
	long long initial_diameter = d.diameter(0); // 0 - 1 - 3: 9
	d.set(e, 1);
	long long updated_diameter = d.diameter(0); // 2 - 1 - 3: 7
	std::cout << lo << ' ' << hi << ' ' << s << '\n';
	std::cout << initial_diameter << ' ' << updated_diameter << '\n';
}

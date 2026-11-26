#include <atl/ds/top_clusters/sum.hpp>
#include <atl/ds/top_tree_engine.hpp>
#include <iostream>
#include <vector>

int main() {
	atl::sum_top_tree<long long> t(std::vector<long long>{5, 1, 4, 2, 3});
	t.assign_parents(std::vector<atl::uint32>{atl::graph::none, 0, 0, 1, 1});
	t.subtree_add(1, 7);
	std::cout << t.subtree_sum(0) << ' ' << t.subtree_sum(1) << '\n';
	t.make_root(4);
	std::cout << t.subtree_sum(0) << '\n';
}

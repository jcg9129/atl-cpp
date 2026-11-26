#include <atl/ds/euler_tour_tree_core.hpp>
#include <functional>
#include <iostream>
#include <vector>

struct sized_sum {
	long long sum = 0, count = 0;
};

struct add_sums {
	sized_sum operator()(sized_sum a, sized_sum b) const { return {a.sum + b.sum, a.count + b.count}; }
};

struct add_to_each {
	sized_sum operator()(long long c, sized_sum a) const { return {a.sum + c * a.count, a.count}; }
};

int main() {
	atl::euler_tour_tree forest(5);
	forest.link(0, 1);
	bool same = forest.connected(1, 0);
	std::size_t k = forest.component_size(0);

	std::vector<sized_sum> start(5, {1, 1});
	atl::euler_tour_tree tree(start, sized_sum{}, 0LL, add_sums{}, add_to_each{}, std::plus<>{});
	tree.assign_parents(std::vector<atl::uint32>{atl::graph::none, 0, 1, 1, 0});
	tree.subtree_apply(1, 0, 10);			   // +10 on 1, 2, 3: the side of 1 away from 0
	long long s = tree.subtree_fold(0, 1).sum; // 0 and 4: 2
	long long all = tree.component_fold(4).sum;
	tree.cut(1, 0);
	std::cout << s << ' ' << all << ' ' << tree.component_size(2) << '\n';
	return same && k == 2 ? 0 : 1;
}

#include <atl/ds/top_clusters/sum.hpp>
#include <atl/ds/top_tree_engine.hpp>
#include <iostream>
#include <vector>

int main() {
	atl::sum_top_tree<long long> t(std::vector<long long>{5, 1, 4, 2, 3});
	t.link(0, 1);
	t.link(0, 2);
	t.link(1, 3);
	t.link(1, 4);
	t.path_add(3, 2, 10);
	std::cout << t.path_sum(2, 3) << '\n';
	t.cut(1, 4);
	t.link(4, 2);
	std::cout << t.path_sum(4, 3) << '\n';
}

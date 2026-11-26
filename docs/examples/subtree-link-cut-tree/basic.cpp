#include <atl/ds/subtree_link_cut_tree_core.hpp>
#include <functional>
#include <iostream>
#include <vector>

int main() {
	atl::subtree_link_cut_tree tree(std::vector<long long>{1, 2, 4, 8}, 0LL, std::plus<>{}, std::negate<>{});
	tree.link(0, 1);
	tree.link(2, 1);
	tree.link(3, 1);
	std::cout << tree.subtree_fold(1) << ' ' << tree.subtree_fold(1, 2) << '\n';
	tree.make_root(0);
	tree.update(3, [](long long value) { return value + 16; });
	std::cout << tree.subtree_fold(1) << ' ' << tree.fold(0, 3) << '\n';
	tree.cut(1, 2);
	std::cout << tree.component_size(0) << ' ' << tree.subtree_fold(2) << '\n';
}

#include <atl/ds/link_cut_tree.hpp>
#include <functional>
#include <iostream>
#include <vector>

int main() {
	atl::link_cut_tree tree(std::vector<long long>{1, 2, 4, 8}, 0LL, std::plus<>{});
	tree.link(0, 1);
	tree.link(2, 1);
	tree.link(3, 2);
	std::cout << tree.fold(0, 3) << ' ' << tree.fold(3, 0) << '\n';
	tree.make_root(0);
	std::cout << tree.parent(1) << ' ' << tree.depth(3) << '\n';
	tree.cut(1, 2);
	std::cout << tree.connected(0, 3) << ' ' << tree.try_fold(2, 3).value() << '\n';
}

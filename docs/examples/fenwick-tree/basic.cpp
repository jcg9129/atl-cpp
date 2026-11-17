#include <atl/ds/binary_indexed_tree.hpp>
#include <array>
#include <iostream>

int main() {
	auto tree = atl::make_binary_indexed_tree_from(std::array{3, 1, 4, 1, 5});
	tree.update(2, 1);
	std::cout << tree.prefix_fold(3) << ' ' << tree.max_right([](int sum) { return sum <= 10; }) << '\n';
	auto grid = atl::make_binary_indexed_tree(0, 3, 4);
	grid.update(7, 1, 2);
	std::cout << grid.prefix_fold(2, 3) << '\n';
}

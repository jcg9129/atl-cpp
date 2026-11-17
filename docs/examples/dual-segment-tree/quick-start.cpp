#include <atl/ds/dual_segment_tree.hpp>
#include <array>
#include <functional>
#include <iostream>

int main() {
	atl::dual_segment_tree library(atl::commutative_tags, std::array{1, 2, 3}, 0, std::plus<>{}, std::plus<>{});
	library.apply(0, 2, 5);
	library.set(2, 9);
	for (int value : library.values()) std::cout << value << '\n';
}

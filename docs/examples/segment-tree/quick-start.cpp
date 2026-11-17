#include <atl/ds/segment_tree.hpp>
#include <array>
#include <functional>
#include <iostream>

int main() {
	atl::segment_tree library(std::array{4, 2, 7, 1, 9}, 0, std::plus<>{});
	library.update(0, [](int x) { return x + 3; });
	library.set(4, 5);
	std::cout << library.fold(0, 5) << ' ' << library.max_right(0, [](int sum) { return sum <= 9; }) << '\n';
}

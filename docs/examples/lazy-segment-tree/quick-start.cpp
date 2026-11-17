#include <atl/ds/lazy_segment_tree.hpp>
#include <array>
#include <functional>
#include <iostream>

int main() {
	struct item {
		int sum, width;
	};

	atl::lazy_segment_tree library(std::array<item, 3>{{{1, 1}, {2, 1}, {3, 1}}}, item{0, 0}, 0, [](item a, item b) { return item{a.sum + b.sum, a.width + b.width}; }, [](int tag, item value) { return item{value.sum + tag * value.width, value.width}; }, std::plus<>{});
	library.apply(1, 3, 5);
	std::cout << library.fold().sum << ' ' << library.at(1).sum << '\n';
}

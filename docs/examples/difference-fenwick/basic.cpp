#include <atl/ds/difference_fenwick.hpp>
#include <array>
#include <iostream>

int main() {
	atl::difference_fenwick<int> values(std::array{1, 2, 3, 4}, 0);
	values.apply(1, 4, 5);
	values.apply(2, -1);
	for (std::size_t i = 0; i < values.size(); ++i)
		std::cout << values.at(i) << (i + 1 == values.size() ? '\n' : ' ');
}

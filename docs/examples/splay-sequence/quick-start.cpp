#include <atl/ds/splay_sequence.hpp>
#include <functional>
#include <iostream>
#include <vector>

int main() {
	atl::reversible_splay_sequence<int, std::plus<>> values(std::vector{1, 2, 3, 4}, 0, std::plus<>{});
	values.insert(2, 5);
	values.reverse(1, 4);
	std::cout << values.fold(1, 4) << '\n';
	bool is_first = true;
	values.for_each([&](int value) {
		if (!is_first) std::cout << ' ';
		is_first = false;
		std::cout << value;
	});
	std::cout << '\n';
}

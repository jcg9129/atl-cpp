#include <atl/ds/persistent_array.hpp>
#include <iostream>
#include <utility>

int main() {
	atl::persistent_array<int> base(3, 0);
	auto const left = base.set(1, 7);
	auto const right = base.set(1, 9);
	std::cout << base[1] << ' ' << left[1] << ' ' << right[1] << '\n';
	char const *separator = "";
	for (int value : left) std::cout << std::exchange(separator, " ") << value;
	std::cout << '\n';

	atl::semi_persistent_array<int> current(3, 0);
	auto const old = current;
	current = std::move(current).set(1, 5);
	current = std::move(current).set(2, 6);
	std::cout << old[1] << ' ' << current[1] << ' ' << current[2] << '\n';
}

#include <atl/ds/persistent_segment_tree.hpp>
#include <functional>
#include <iostream>
#include <vector>

int main() {
	atl::persistent_segment_tree a(std::vector{1, 2, 3}, 0, std::plus<>{});
	auto b = a.update(1, [](int x) { return x + 5; });
	auto c = b.copy_range(a, 0, 2);
	std::cout << a.fold() << ' ' << b.fold() << ' ' << c.fold() << '\n';
}

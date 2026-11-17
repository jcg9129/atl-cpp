#include <atl/ds/persistent_treap_sequence.hpp>
#include <functional>
#include <iostream>
#include <vector>

int main() {
	atl::persistent_treap_sequence a(std::vector{1, 2, 3}, 0, std::plus<>{});
	auto b = a.insert(1, 9);
	auto [left, right] = b.split(2);
	auto joined = right.concat(left);
	std::cout << a.fold() << ' ' << b.fold() << ' ' << joined.at(0) << ' ' << joined.fold() << '\n';
}

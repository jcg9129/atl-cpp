#include <atl/ds/treap_map.hpp>
#include <functional>
#include <iostream>

int main() {
	atl::treap_map<int, long long, std::ranges::less, std::plus<>> values(0LL, std::plus<>{});
	values.insert_or_assign(7, 70);
	values.update(9, [](long long value) { return value + 5; });
	std::cout << values.fold(0, 10) << ' ' << values.rank(9) << '\n';
}

#include <atl/ds/persistent_list.hpp>
#include <iostream>

int main() {
	atl::persistent_list<int> base;
	auto a = base.push_front(1).push_front(2).push_front(3);
	auto b = a.set(1, 9).pop_front();
	std::cout << a.front() << ' ' << a[1] << ' ' << b.front() << '\n';
}

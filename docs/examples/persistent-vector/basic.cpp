#include <atl/ds/persistent_vector.hpp>
#include <iostream>
#include <vector>

int main() {
	atl::persistent_vector a(std::vector{1, 2, 3});
	auto b = a.set(0, 9);
	auto c = a.concat(b.slice(1, 3));
	std::cout << a[0] << ' ' << b[0] << ' ' << c.size() << ' ' << c.back() << '\n';
	atl::semi_persistent_vector<int> linear;
	linear = linear.push_back(4);
	auto snapshot = linear;
	linear = linear.push_back(5);
	std::cout << snapshot.back() << ' ' << linear.back() << '\n';
}

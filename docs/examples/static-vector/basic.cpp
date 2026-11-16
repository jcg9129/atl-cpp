#include <atl/ds/static_vector.hpp>
#include <iostream>

int main() {
	atl::static_vector<int, 4> values{2, 3};
	values.emplace_back(5);
	values.pop_back();
	values.resize(3, 7);
	for (int value : values) std::cout << value << '\n';
}

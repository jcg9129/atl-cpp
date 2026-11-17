#include <atl/ds/direct_rmq.hpp>
#include <array>
#include <iostream>

int main() {
	std::array const values{5, 2, 8, 1, 1, 3};
	atl::direct_rmq query(values);
	std::cout << query.position(0, 5) << ' ' << query.fold(0, 5) << '\n';
}

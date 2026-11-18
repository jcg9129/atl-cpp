#include <atl/nt/discrete_log.hpp>
#include <iostream>

int main() {
	std::cout << *atl::discrete_log(3, 13, 17) << '\n';
	std::cout << atl::discrete_log(2, 3, 7).has_value() << '\n';
	std::cout << *atl::discrete_log(6, 0, 12) << '\n';
	std::cout << atl::multiplicative_order(2, 7) << ' ' << atl::primitive_root(17) << '\n';
}

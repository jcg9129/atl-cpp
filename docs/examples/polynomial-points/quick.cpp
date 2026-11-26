#include <atl/nt/mod_int_core.hpp>
#include <atl/nt/polynomial_points_native.hpp>
#include <iostream>
#include <vector>

int main() {
	using coefficient = atl::mod_int32<998244353>;
	std::vector<coefficient> const polynomial{1, 2, 3}, points{4, 5, 6};

	atl::polynomial_point_tree<coefficient> const tree(points);
	auto const values = tree.evaluate(polynomial);
	for (char const *separator = ""; auto const value : values) {
		std::cout << separator << value.value();
		separator = " ";
	}
	std::cout << '\n';

	for (char const *separator = ""; auto const value : tree.interpolate(values)) {
		std::cout << separator << value.value();
		separator = " ";
	}
	std::cout << '\n';

	for (char const *separator = ""; auto const value : atl::polynomial_taylor_shift(polynomial, coefficient(1))) {
		std::cout << separator << value.value();
		separator = " ";
	}
	std::cout << '\n';

	for (char const *separator = ""; auto const value : atl::polynomial_from_roots(points)) {
		std::cout << separator << value.value();
		separator = " ";
	}
	std::cout << '\n';
}

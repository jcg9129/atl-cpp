#include <atl/nt/mod_int_static.hpp>
#include <atl/nt/polynomial_points_ntt.hpp>
#include <iostream>
#include <vector>

int main() {
	using coefficient = atl::mod_int32<998244353>;
	std::vector<coefficient> const polynomial{1, 2, 3};

	for (char const *separator = ""; auto const value : atl::polynomial_geometric_evaluation(polynomial, coefficient(1), coefficient(2), 4)) {
		std::cout << separator << value.value();
		separator = " ";
	}
	std::cout << '\n';

	std::vector<coefficient> const samples{1, 6, 17};
	for (char const *separator = ""; auto const value : atl::sample_point_shift(samples, coefficient(3), 4)) {
		std::cout << separator << value.value();
		separator = " ";
	}
	std::cout << '\n';

	std::vector<std::vector<coefficient>> const factors{{1, 1}, {1, 2}, {3}};
	for (char const *separator = ""; auto const value : atl::polynomial_product(factors)) {
		std::cout << separator << value.value();
		separator = " ";
	}
	std::cout << '\n';
}

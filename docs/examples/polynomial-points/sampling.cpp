#include <atl/nt/mod_int.hpp>
#include <atl/nt/polynomial_points.hpp>
#include <iostream>
#include <vector>

int main() {
	using coefficient = atl::mod_int32<998244353>;
	std::vector<coefficient> const polynomial{1, 2, 3};

	for (auto const value : atl::polynomial_geometric_evaluation(polynomial, coefficient(1), coefficient(2), 4)) std::cout << value.value() << ' ';
	std::cout << '\n';

	std::vector<coefficient> const samples{1, 6, 17};
	for (auto const value : atl::sample_point_shift(samples, coefficient(3), 4)) std::cout << value.value() << ' ';
	std::cout << '\n';

	std::vector<std::vector<coefficient>> const factors{{1, 1}, {1, 2}, {3}};
	for (auto const value : atl::polynomial_product(factors)) std::cout << value.value() << ' ';
	std::cout << '\n';
}

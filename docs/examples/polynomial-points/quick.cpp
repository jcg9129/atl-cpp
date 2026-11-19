#include <atl/nt/mod_int.hpp>
#include <atl/nt/polynomial_points.hpp>
#include <iostream>
#include <vector>

int main() {
	using coefficient = atl::mod_int32<998244353>;
	std::vector<coefficient> const polynomial{1, 2, 3}, points{4, 5, 6};

	atl::polynomial_point_tree<coefficient> const tree(points);
	auto const values = tree.evaluate(polynomial);
	for (auto const value : values) std::cout << value.value() << ' ';
	std::cout << '\n';

	for (auto const value : tree.interpolate(values)) std::cout << value.value() << ' ';
	std::cout << '\n';

	for (auto const value : atl::polynomial_taylor_shift(polynomial, coefficient(1))) std::cout << value.value() << ' ';
	std::cout << '\n';

	for (auto const value : atl::polynomial_from_roots(points)) std::cout << value.value() << ' ';
	std::cout << '\n';
}

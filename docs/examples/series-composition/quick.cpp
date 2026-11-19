#include <atl/nt/mod_int.hpp>
#include <atl/nt/series_composition.hpp>
#include <iostream>
#include <vector>

int main() {
	using coefficient = atl::mod_int32<998244353>;
	std::vector<coefficient> const outer{1, 2, 3}, inner{0, 1, 1};

	for (auto const value : atl::series_composition(outer, inner, 5)) std::cout << value.value() << ' ';
	std::cout << '\n';

	auto const inverse = atl::series_compositional_inverse(inner, 5);
	for (auto const value : inverse) std::cout << value.value() << ' ';
	std::cout << '\n';

	for (auto const value : atl::series_composition(inner, inverse, 5)) std::cout << value.value() << ' ';
	std::cout << '\n';

	for (auto const value : atl::series_power_projection(outer, inner, 5)) std::cout << value.value() << ' ';
	std::cout << '\n';
}

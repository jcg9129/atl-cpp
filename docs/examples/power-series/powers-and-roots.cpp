#include <atl/nt/mod_int_static.hpp>
#include <atl/nt/power_series_ntt.hpp>
#include <iostream>
#include <vector>

int main() {
	using coefficient = atl::mod_int32<998244353>;
	std::vector<coefficient> const binomial{1, 1};
	for (char const *separator = ""; auto const value : atl::series_power(binomial, 1000000000000000000ull, 4)) {
		std::cout << separator << value.value();
		separator = " ";
	}
	std::cout << '\n';

	std::vector<coefficient> const square{0, 0, 4, 4, 1};
	if (auto const root = atl::series_square_root(square, 7))
		for (char const *separator = ""; auto const value : *root) {
			std::cout << separator << value.value();
			separator = " ";
		}
	std::cout << '\n';

	auto const division = atl::polynomial_division(std::vector<coefficient>{1, 3, 3, 1}, std::vector<coefficient>{1, 2});
	std::cout << division.quotient.size() << ' ' << division.remainder.size() << '\n';

	std::vector<coefficient> const sparse{1, 0, 0, 0, 0, 0, 0, 0, 1};
	for (char const *separator = ""; auto const value : atl::series_inverse_sparse(sparse, 10)) {
		std::cout << separator << value.value();
		separator = " ";
	}
	std::cout << '\n';
}

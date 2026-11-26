#include <atl/nt/mod_int_static.hpp>
#include <atl/nt/polynomial_gcd_ntt.hpp>
#include <iostream>
#include <vector>

int main() {
	using coefficient = atl::mod_int32<998244353>;
	std::vector<coefficient> const first{2, 3, 1}, second{3, 4, 1};

	for (char const *separator = ""; auto const value : atl::polynomial_gcd(first, second)) {
		std::cout << separator << value.value();
		separator = " ";
	}
	std::cout << '\n';

	auto const identity = atl::polynomial_extended_gcd(first, second);
	for (auto const value : identity.first_cofactor) std::cout << value.value() << ' ';
	std::cout << '|';
	for (auto const value : identity.second_cofactor) std::cout << ' ' << value.value();
	std::cout << '\n';

	auto const inverse = atl::polynomial_inverse_mod(std::vector<coefficient>{0, 1}, std::vector<coefficient>{1, 0, 1});
	for (char const *separator = ""; auto const value : *inverse) {
		std::cout << separator << value.value();
		separator = " ";
	}
	std::cout << '\n';

	std::cout << atl::polynomial_inverse_mod(first, second).has_value() << '\n';
}

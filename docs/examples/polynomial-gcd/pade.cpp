#include <atl/nt/mod_int.hpp>
#include <atl/nt/polynomial_gcd.hpp>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
	using coefficient = atl::mod_int32<998244353>;
	auto const print = [](std::vector<coefficient> const &polynomial) {
		for (char const *separator = ""; auto const value : polynomial) {
			std::int64_t const centered = value.value() > coefficient::mod() / 2 ? std::int64_t(value.value()) - coefficient::mod() : value.value();
			std::cout << separator << centered;
			separator = " ";
		}
		std::cout << '\n';
	};

	// The first terms of x / (1 - x - x^2), the Fibonacci numbers; three terms are enough for numerator degree 1 and denominator degree 2.
	std::vector<coefficient> const series{0, 1, 1, 2, 3, 5, 8, 13};
	auto const approximant = atl::pade_approximant(series, 1, 2);
	print(approximant->numerator);
	print(approximant->denominator);

	// A series that no rational function of these degrees matches has no approximant.
	std::cout << atl::pade_approximant(std::vector<coefficient>{0, 1, 0, 1}, 0, 1).has_value() << '\n';
}

#include <atl/nt/linear_recurrence.hpp>
#include <atl/nt/mod_int.hpp>
#include <array>
#include <iostream>

int main() {
	using mint = atl::mod_int32<998244353>;
	std::array<mint, 8> const fib{0, 1, 1, 2, 3, 5, 8, 13};
	auto const coefficients = atl::berlekamp_massey(fib);
	std::cout << coefficients.size() << ' ' << coefficients[0].value() << ' '
			  << coefficients[1].value() << '\n';
	std::cout << atl::linear_recurrence_term(coefficients, fib, 10u).value() << '\n';
	std::array<mint, 1> const numerator{1};
	std::array<mint, 2> const denominator{1, -2};
	std::cout << atl::rational_series_coefficient(numerator, denominator, 10u).value() << '\n';
}

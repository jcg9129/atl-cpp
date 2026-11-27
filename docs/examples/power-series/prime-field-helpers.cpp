#include <atl/nt/mod_int_static.hpp>
#include <atl/nt/prime_field_helpers.hpp>
#include <atl/nt/square_root_mod_core.hpp>
#include <iostream>
#include <vector>

int main() {
	using coefficient = atl::mod_int32<998244353>;
	atl::factorial_table<coefficient> const table(100);
	std::cout << table.binomial(10, 3).value() << '\n';

	std::vector<coefficient> const polynomial{3, 4, 5};
	for (char const *separator = ""; auto const value : atl::series_derivative(polynomial)) {
		std::cout << separator << value.value();
		separator = " ";
	}
	std::cout << '\n';
	for (char const *separator = ""; auto const value : atl::series_integral(polynomial)) {
		std::cout << separator << value.value();
		separator = " ";
	}
	std::cout << '\n';

	for (char const *separator = ""; auto const value : atl::batch_inverse(std::vector<coefficient>{2, 3})) {
		std::cout << separator << (value * coefficient(6)).value();
		separator = " ";
	}
	std::cout << '\n';

	std::cout << atl::square_root_mod(coefficient(4))->value() << ' ' << atl::square_root_mod(4, 17).value() << ' '
			  << atl::square_root_mod(3, 17).has_value() << '\n';
}

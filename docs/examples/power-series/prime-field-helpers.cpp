#include <atl/nt/batch_inverse.hpp>
#include <atl/nt/factorial_table.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/series_calculus.hpp>
#include <atl/nt/square_root_mod.hpp>
#include <iostream>
#include <vector>

int main() {
	using coefficient = atl::mod_int32<998244353>;
	atl::factorial_table<coefficient> const table(100);
	std::cout << table.binomial(10, 3).value() << '\n';

	std::vector<coefficient> const polynomial{3, 4, 5};
	for (auto const value : atl::series_derivative(polynomial)) std::cout << value.value() << ' ';
	std::cout << '\n';
	for (auto const value : atl::series_integral(polynomial)) std::cout << value.value() << ' ';
	std::cout << '\n';

	for (auto const value : atl::batch_inverse(std::vector<coefficient>{2, 3})) std::cout << (value * coefficient(6)).value() << ' ';
	std::cout << '\n';

	std::cout << atl::square_root_mod(coefficient(4))->value() << ' ' << atl::square_root_mod(4, 17).value() << ' '
			  << atl::square_root_mod(3, 17).has_value() << '\n';
}

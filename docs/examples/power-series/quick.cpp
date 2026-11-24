#include <atl/nt/mod_int_core.hpp>
#include <atl/nt/power_series_native.hpp>
#include <iostream>
#include <vector>

int main() {
	using coefficient = atl::mod_int32<998244353>;
	std::vector<coefficient> const unit{1, 1};
	std::vector<coefficient> const zero{0, 1};

	for (char const *separator = ""; auto const value : atl::series_inverse(unit, 5)) {
		std::cout << separator << value.value();
		separator = " ";
	}
	std::cout << '\n';

	auto const logarithm = atl::series_logarithm(unit, 5);
	for (char const *separator = ""; auto const value : atl::series_exponential(logarithm, 5)) {
		std::cout << separator << value.value();
		separator = " ";
	}
	std::cout << '\n';

	for (char const *separator = ""; auto const value : atl::series_exponential(zero, 5)) {
		std::cout << separator << value.value();
		separator = " ";
	}
	std::cout << '\n';
}

#include <atl/nt/mod_int_static.hpp>
#include <atl/nt/online_convolution_ntt.hpp>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
	using coefficient = atl::mod_int32<998244353>;

	// Catalan numbers from C = 1 + x C^2: the coefficient of x^(n+1) is the coefficient of x^n of C * C.
	atl::online_convolution<coefficient> square;
	std::vector<coefficient> catalan{1};
	for (std::size_t n = 0; n < 11; ++n) catalan.push_back(square.push(catalan[n], catalan[n]));
	for (char const *separator = ""; auto const value : catalan) {
		std::cout << separator << value.value();
		separator = " ";
	}
	std::cout << '\n';
}

#include <atl/nt/basic.hpp>
#include <iostream>

int main() {
	std::cout << atl::is_prime(1'000'000'007) << '\n';
	char const *separator = "";
	for (auto const &factor : atl::factorization(360)) {
		std::cout << separator << factor.prime << '^' << factor.exponent;
		separator = " ";
	}
	std::cout << '\n';
}

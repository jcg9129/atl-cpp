#include <atl/nt/sieve.hpp>
#include <iostream>

int main() {
	atl::sieve const table(100);
	std::cout << table.is_prime(97) << ' ' << table.min_prime_factor(84) << '\n';
	char const *separator = "";
	for (auto const &factor : table.factorization(84)) {
		std::cout << separator << factor.prime << '^' << factor.exponent;
		separator = " ";
	}
	std::cout << '\n';
	separator = "";
	for (auto divisor : table.divisors(12)) {
		std::cout << separator << divisor;
		separator = " ";
	}
	std::cout << '\n';
}

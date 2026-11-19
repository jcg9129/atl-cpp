#include <atl/nt/mod_int.hpp>
#include <atl/nt/online_convolution.hpp>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
	using coefficient = atl::mod_int32<998244353>;
	std::size_t const length = 12;

	// Partition numbers from n p(n) = sum over k = 1..n of sigma(k) p(n - k), with sigma the sum of divisors.
	// The kernel holds sigma(k) at index k - 1, so pushing p(n - 1) returns the sum for p(n).
	std::vector<coefficient> sigma(length - 1);
	for (std::size_t k = 1; k < length; ++k)
		for (std::size_t divisor = 1; divisor <= k; ++divisor)
			if (k % divisor == 0) sigma[k - 1] += coefficient(divisor);

	atl::online_fixed_convolution<coefficient> sums(sigma);
	std::vector<coefficient> partitions{1};
	for (std::size_t n = 1; n < length; ++n) partitions.push_back(sums.push(partitions[n - 1]) / coefficient(n));
	for (auto const value : partitions) std::cout << value.value() << ' ';
	std::cout << '\n';
}

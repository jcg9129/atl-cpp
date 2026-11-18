#include <atl/nt/bitwise_convolution.hpp>
#include <atl/nt/mod_int.hpp>
#include <iostream>
#include <vector>

int main() {
	using Z = atl::mod_int<atl::uint32, 998244353>;
	std::vector<Z> a{1, 2, 3, 4}, b{5, 6, 7, 8};
	std::cout << atl::or_convolution(a, b)[3].value() << '\n';
	std::cout << atl::and_convolution(a, b)[3].value() << '\n';
	std::cout << atl::xor_convolution(a, b)[0].value() << '\n';
	std::cout << atl::subset_convolution(a, b)[3].value() << '\n';
	atl::subset_zeta(a);
	std::cout << a[3].value() << '\n';
}

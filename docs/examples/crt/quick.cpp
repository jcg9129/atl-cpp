#include <atl/nt/crt.hpp>
#include <array>
#include <iostream>

int main() {
	auto const linear = atl::solve_linear_congruence(6, 4, 10u);
	std::cout << linear->residue << ' ' << linear->modulus << '\n';
	std::array const system{atl::congruence<unsigned>{2, 3},
							atl::congruence<unsigned>{3, 5}, atl::congruence<unsigned>{2, 7}};
	auto const combined = atl::chinese_remainder(system);
	std::cout << combined->residue << ' ' << combined->modulus << '\n';
	std::cout << atl::chinese_remainder_mod(system, 10u) << '\n';
}

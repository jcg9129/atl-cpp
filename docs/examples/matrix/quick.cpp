#include <atl/nt/bit_matrix.hpp>
#include <atl/nt/gaussian_elimination.hpp>
#include <atl/nt/matrix.hpp>
#include <atl/nt/mod_int.hpp>
#include <iostream>

int main() {
	using mint = atl::mod_int32<998244353>;
	atl::matrix<mint> const fib{{1, 1}, {1, 0}};
	std::cout << atl::power(fib, 10u)(0, 1).value() << '\n';
	atl::matrix<mint> const a{{2, 1, 1}, {1, 3, 2}, {1, 0, 0}};
	std::cout << atl::determinant(a).value() << '\n';
	atl::bit_matrix bits(2, 3);
	bits.set(0, 0);
	bits.set(0, 1);
	bits.set(1, 1);
	bits.set(1, 2);
	auto const solution = atl::solve_linear_system(bits, {true, false});
	std::cout << atl::rank(bits) << ' ' << solution.is_solvable << '\n';
}

#include <atl/ds/weighted_dsu.hpp>
#include <atl/nt/mod_int.hpp>
#include <iostream>

using residue = atl::mod_int32<998244353>;

/** @brief A determinant-one matrix over the modular field. */
struct matrix {
	residue a, b, c, d;
	bool operator==(matrix const &) const = default;
};

/** @brief Compose potentials by matrix multiplication, preserving operand order. */
struct multiply {
	matrix operator()(matrix const &x, matrix const &y) const {
		return {x.a * y.a + x.b * y.c, x.a * y.b + x.b * y.d, x.c * y.a + x.d * y.c, x.c * y.b + x.d * y.d};
	}
};

/** @brief The adjugate is the inverse because the determinant is one. */
struct inverse {
	matrix operator()(matrix const &x) const { return {x.d, -x.b, -x.c, x.a}; }
};

int main() {
	atl::weighted_dsu potentials(3, matrix{1, 0, 0, 1}, multiply{}, inverse{});
	potentials.unite(0, 1, matrix{1, 2, 0, 1});
	potentials.unite(1, 2, matrix{1, 0, 3, 1});
	auto const result = *potentials.difference(0, 2);
	std::cout << result.a << ' ' << result.b << '\n';
	std::cout << result.c << ' ' << result.d << '\n';
}

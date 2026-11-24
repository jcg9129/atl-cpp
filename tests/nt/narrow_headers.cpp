#include <atl/nt/mod_int_core.hpp>
#include <atl/nt/series_composition_native.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <vector>

TEST_CASE("native headers: transform paths without optional implementations", "[power_series][series_composition]") {
	using C = atl::mod_int32<998244353>;
	using V = std::vector<C>;
	std::size_t const n = 129;
	REQUIRE(atl::series_inverse(V{1, -1}, n) == V(n, C(1)));

	V const f{0, 1, 1};
	auto const g = atl::series_compositional_inverse(f, n);
	// Independently check g + g^2 = x by a truncated schoolbook product.
	V composed = g;
	for (std::size_t i = 0; i < n; ++i)
		for (std::size_t j = 0; j < n - i; ++j) composed[i + j] += g[i] * g[j];
	V identity(n);
	identity[1] = C(1);
	REQUIRE(composed == identity);
	REQUIRE(atl::series_composition(f, g, n) == identity);
}

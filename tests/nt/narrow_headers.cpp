#include <atl/nt/mod_int_core.hpp>
#include <atl/nt/series_composition_native.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <vector>

TEST_CASE("native headers: transform paths without optional implementations", "[power_series][series_composition]") {
	using C = atl::mod_int32<998244353>;
	using V = std::vector<C>;
	std::size_t const n = 129;
	REQUIRE((atl::series_inverse(V{1, -1}, n) == V(n, C(1))));

	V const f{0, 1, 1};
	auto const g = atl::series_compositional_inverse(f, n);
	// Independently check g + g^2 = x by a truncated schoolbook product.
	V composed = g;
	for (std::size_t i = 0; i < n; ++i)
		for (std::size_t j = 0; j < n - i; ++j) composed[i + j] += g[i] * g[j];
	V identity(n);
	identity[1] = C(1);
	REQUIRE((composed == identity));
	REQUIRE((atl::series_composition(f, g, n) == identity));
}

template<atl::uint32 Mod>
void check_fallback_series() {
	using C = atl::mod_int32<Mod>;
	std::size_t const n = 257;
	std::vector<C> f(129), expected(n);
	f[0] = 1;
	for (std::size_t i = 1; i < f.size(); ++i) f[i] = C(i * i + 17 * i + 3);
	expected[0] = 1;
	// Coefficient recurrence from f * inverse(f) = 1, independent of transforms.
	for (std::size_t k = 1; k < n; ++k)
		for (std::size_t i = 1; i < f.size() && i <= k; ++i) expected[k] -= f[i] * expected[k - i];
	REQUIRE((atl::series_inverse(f, n) == expected));
	std::vector<C> product(f.size() * 2 - 1);
	for (std::size_t i = 0; i < f.size(); ++i)
		for (std::size_t j = 0; j < f.size(); ++j) product[i + j] += f[i] * f[j];
	REQUIRE((atl::convolution(f, f) == product));
}

TEST_CASE("native series retains FFT split and CRT fallbacks", "[power_series]") {
	check_fallback_series<1000000007>(); // Non-NTT prime below 2^30: FFT split.
	check_fallback_series<4294967291>(); // Prime above the FFT bound: CRT.
}

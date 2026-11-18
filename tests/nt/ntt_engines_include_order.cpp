// clang-format off
// The engine headers come first: they must not depend on an earlier include.
#include <atl/nt/detail/ntt_simd.hpp>
#include <atl/nt/detail/ntt_conv8.hpp>
#include <algorithm>
#include <cstddef>
#include <vector>
#include <atl/nt/convolution.hpp>
#include <catch2/catch.hpp>
// clang-format on

TEST_CASE("ntt engines: engine-first includes and linked instantiations", "[nt]") {
	using Z = atl::mod_int<atl::uint32, 998244353>;
	std::vector<Z> a(64, Z(1)), b(64, Z(2)), expected(127);
	for (std::size_t i = 0; i < expected.size(); ++i)
		expected[i] = Z(2 * (i < 64 ? i + 1 : 127 - i));
	REQUIRE(atl::convolution<atl::scalar_ntt>(a, b) == expected);
	REQUIRE(atl::convolution<atl::simd_ntt>(a, b) == expected);
	REQUIRE(atl::convolution<atl::conv8_ntt>(a, b) == expected);
}

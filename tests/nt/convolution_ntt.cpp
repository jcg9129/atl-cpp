#include <atl/nt/convolution_ntt.hpp>
#include <atl/nt/polynomial_gcd_ntt.hpp>
#include <atl/nt/series_composition_ntt.hpp>
#include <catch2/catch.hpp>
#include <cstddef>
#include <utility>
#include <vector>
#include "../expect_abort.hpp"

TEST_CASE("convolution_ntt: balanced, square and overlap-add match schoolbook", "[convolution]") {
	using C = atl::mod_int32<998244353>;
	for (auto const &[n, m] : {std::pair{0, 70}, {7, 13}, {73, 97}, {65, 1200}}) {
		std::vector<C> left(n), right(m);
		for (int i = 0; i < n; ++i) left[i] = C(i * i - 3 * i + 1);
		for (int i = 0; i < m; ++i) right[i] = C(7 * i + 2);
		std::vector<C> expected(n && m ? n + m - 1 : 0);
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < m; ++j) expected[i + j] += left[i] * right[j];
		CHECK((atl::convolution_ntt(left, right) == expected));
	}
	std::vector<C> const a(71, C(3));
	auto const squared = atl::convolution_ntt(a, a);
	for (std::size_t i = 0; i < squared.size(); ++i)
		CHECK((squared[i] == C(9 * (i < a.size() ? i + 1 : squared.size() - i))));
}

TEST_CASE("convolution_ntt: order precondition also applies below the naive cutoff", "[convolution]") {
	using C = atl::mod_int32<17>;
	std::vector<C> const valid(8, 1), invalid(9, 1);
	CHECK(atl::convolution_ntt(valid, valid).size() == 15);
	CHECK(atl_test::aborts([&] { (void)atl::convolution_ntt(invalid, invalid); }));
}

TEST_CASE("NTT-only series and polynomial layers run without full extensions", "[power_series]") {
	using C = atl::mod_int32<998244353>;
	using V = std::vector<C>;
	CHECK((atl::series_inverse(V{1, -1}, 129) == V(129, C(1))));
	auto const root = atl::series_square_root(V{1, 2, 1}, 129);
	REQUIRE(root.has_value());
	CHECK(((*root)[0] == C(1)));
	CHECK(((*root)[1] == C(1)));
	CHECK((atl::polynomial_gcd(V{2, 3, 1}, V{3, 4, 1}) == V{1, 1}));
	V const f{0, 1, 1};
	auto const inverse = atl::series_compositional_inverse(f, 129);
	V identity(129);
	identity[1] = C(1);
	CHECK((atl::series_composition(f, inverse, 129) == identity));
}

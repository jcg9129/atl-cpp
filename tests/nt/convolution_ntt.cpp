#include <atl/nt/convolution_ntt.hpp>
#include <atl/nt/online_convolution_ntt.hpp>
#include <atl/nt/polynomial_gcd_ntt.hpp>
#include <atl/nt/prime_field_helpers.hpp>
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

TEST_CASE("NTT-only online streams match defining sums without full extensions", "[online_convolution]") {
	using C = atl::mod_int32<998244353>;
	std::vector<C> left(257), right(193);
	for (std::size_t i = 0; i < left.size(); ++i) left[i] = C(i * i + 3);
	for (std::size_t i = 0; i < right.size(); ++i) right[i] = C(7 * i - 2);
	atl::online_convolution<C> product;
	atl::online_fixed_convolution<C> fixed(right);
	product.reserve(left.size());
	fixed.reserve(left.size());
	for (std::size_t k = 0; k < left.size(); ++k) {
		C expected(0);
		for (std::size_t j = 0; j <= k && j < right.size(); ++j) expected += left[k - j] * right[j];
		CHECK((product.push(left[k], k < right.size() ? right[k] : C(0)) == expected));
		CHECK((fixed.push(left[k]) == expected));
	}
}

TEST_CASE("static prime helpers work without runtime arithmetic", "[power_series]") {
	using C = atl::mod_int32<17>;
	atl::factorial_table<C> const table(16);
	CHECK((table.binomial(10, 3) == C(120)));
	CHECK(((atl::batch_inverse(std::vector<C>{2, 3}) == std::vector<C>{9, 6})));
	for (int value = 0; value < 17; ++value) {
		auto const root = atl::square_root_mod(C(value));
		bool found = false;
		for (int candidate = 0; candidate <= 8; ++candidate) {
			if (candidate * candidate % 17 != value) continue;
			CHECK((root == C(candidate)));
			found = true;
			break;
		}
		CHECK((root.has_value() == found));
	}
	CHECK((atl::square_root_mod(atl::mod_int32<2>(1)) == atl::mod_int32<2>(1)));
	using Wide = atl::mod_int64<18446744073709551557ull>;
	CHECK((atl::square_root_mod(Wide(4)) == Wide(2)));
	std::vector<C> const polynomial{3, 4, 5};
	CHECK(((atl::series_derivative(polynomial) == std::vector<C>{4, 10})));
	CHECK(((atl::series_integral(polynomial) == std::vector<C>{0, 3, 2, 13})));
}

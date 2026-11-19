#include <atl/experimental/nt/series_composition_variants.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <atl/nt/series_composition.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <random>
#include <vector>
#include "../expect_abort.hpp"

namespace {
	/** @brief Scalar engine that records the length of every transform; a fused convolution records two forwards and one inverse. */
	struct counting_engine {
		static inline std::vector<std::size_t> forwards, inverses;

		template<atl::static_mod_int Coefficient>
		static consteval bool supports() { return atl::scalar_ntt::supports<Coefficient>(); }

		template<atl::static_mod_int Coefficient>
		static void forward(decltype(Coefficient::mod()) *data, std::uint32_t height) {
			forwards.push_back(std::size_t{1} << height);
			atl::scalar_ntt::forward<Coefficient>(data, height);
		}

		template<atl::static_mod_int Coefficient>
		static void inverse(decltype(Coefficient::mod()) *data, std::uint32_t height) {
			inverses.push_back(std::size_t{1} << height);
			atl::scalar_ntt::inverse<Coefficient>(data, height);
		}

		template<atl::static_mod_int Coefficient>
		static void convolve(decltype(Coefficient::mod()) *left, decltype(Coefficient::mod()) *right, std::uint32_t height) {
			forwards.push_back(std::size_t{1} << height);
			forwards.push_back(std::size_t{1} << height);
			inverses.push_back(std::size_t{1} << height);
			atl::scalar_ntt::convolve<Coefficient>(left, right, height);
		}
	};

	/** @brief An engine that supports no modulus, so the library must take its fallback paths and never call the counting callbacks. */
	struct unsupported_engine : counting_engine {
		template<atl::static_mod_int Coefficient>
		static consteval bool supports() { return false; }
	};

	/** @brief Independent schoolbook truncated product, sharing no kernel with the code under test. */
	template<atl::static_prime_mod_int Coefficient>
	std::vector<Coefficient> oracle_multiply(std::vector<Coefficient> const &a, std::vector<Coefficient> const &b, std::size_t n) {
		std::vector<Coefficient> c(n);
		for (std::size_t i = 0; i < a.size() && i < n; ++i)
			if (a[i] != Coefficient(0))
				for (std::size_t j = 0; j < b.size() && j < n - i; ++j) c[i + j] += a[i] * b[j];
		return c;
	}

	/** @brief Independent finite-polynomial Horner oracle, including nonzero inner constants. */
	template<atl::static_prime_mod_int Coefficient>
	std::vector<Coefficient> oracle_compose(std::vector<Coefficient> const &f, std::vector<Coefficient> const &g, std::size_t n) {
		std::vector<Coefficient> result(n);
		if (n == 0) return result;
		for (std::size_t i = f.size(); i > 0; --i) {
			result = oracle_multiply(result, g, n);
			result[0] += f[i - 1];
		}
		return result;
	}

	/** @brief Independent repeated-power and scalar-dot-product oracle. */
	template<atl::static_prime_mod_int Coefficient>
	std::vector<Coefficient> oracle_project(std::vector<Coefficient> const &w, std::vector<Coefficient> const &g, std::size_t count) {
		std::vector<Coefficient> result(count), power(w.size());
		if (w.empty()) return result;
		power[0] = Coefficient(1);
		for (std::size_t i = 0; i < count; ++i) {
			for (std::size_t j = 0; j < w.size(); ++j) result[i] += w[j] * power[j];
			power = oracle_multiply(power, g, w.size());
		}
		return result;
	}

	/** @brief Taylor oracle from Pascal's triangle, valid even when factorials vanish. */
	template<atl::static_prime_mod_int Coefficient>
	std::vector<Coefficient> oracle_taylor(std::vector<Coefficient> const &f, std::vector<Coefficient> g, std::size_t n) {
		if (n == 0) return {};
		Coefficient const constant = g.empty() ? Coefficient(0) : g[0];
		if (!g.empty()) g[0] = Coefficient(0);
		std::vector<Coefficient> result(n), power(n);
		power[0] = Coefficient(1);
		for (std::size_t k = 0; k < f.size(); ++k) {
			Coefficient derivative(0);
			std::vector<Coefficient> row{Coefficient(1)};
			for (std::size_t degree = 0; degree < f.size(); ++degree) {
				if (degree >= k) derivative += f[degree] * row[k] * constant.power(degree - k);
				row.push_back(Coefficient(1));
				for (std::size_t j = row.size() - 2; j > 0; --j) row[j] += row[j - 1];
			}
			for (std::size_t j = 0; j < n; ++j) result[j] += derivative * power[j];
			power = oracle_multiply(power, g, n);
		}
		return result;
	}

	/** @brief Check every composition and projection contender across characteristic and structural cases. */
	template<atl::static_prime_mod_int Coefficient>
	void check_family() {
		using Vector = std::vector<Coefficient>;
		using Binary = Vector (*)(Vector const &, Vector const &, std::size_t);
		std::mt19937_64 random(7002);
		for (std::size_t n : {0, 1, 2, 3, 4, 7, 8, 9, 13, 15, 16, 17, 23, 24, 25, 31, 32, 33, 47, 48, 49, 63, 64, 65}) {
			CAPTURE(n, Coefficient::mod());
			Vector outer(n + 3), weights(n), inner(n + 2);
			for (auto &value : outer) value = Coefficient(random());
			for (auto &value : weights) value = Coefficient(random());
			for (auto &value : inner) value = Coefficient(random());
			for (std::size_t shape = 0; shape < 8; ++shape) {
				Vector g = inner;
				if (shape < 3)
					for (std::size_t i = 0; i <= shape && i < g.size(); ++i) g[i] = Coefficient(0);
				if (shape == 3) std::ranges::fill(g, Coefficient(0));
				if (shape >= 4) {
					std::ranges::fill(g, Coefficient(0));
					g[0] = Coefficient(3);
					g[shape - 3 < g.size() ? shape - 3 : 1] = Coefficient(1);
				}
				auto const expected = oracle_compose(outer, g, n);
				for (Binary solver : {atl::series_composition<atl::default_ntt_engine, Vector, Vector>,
									  atl::experimental::series_composition_by_brent_kung<atl::default_ntt_engine, Vector, Vector>,
									  atl::experimental::series_composition_by_fast_horner<atl::default_ntt_engine, Vector, Vector>,
									  atl::experimental::series_composition_by_schoolbook_horner<atl::default_ntt_engine, Vector, Vector>})
					REQUIRE(solver(outer, g, n) == expected);
				REQUIRE(atl::series_composition(Vector{}, g, n) == Vector(n));
				if (n <= 17) REQUIRE(oracle_taylor(outer, g, n) == expected);
				for (std::size_t count : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{7}, std::size_t{8}, std::size_t{9}, n + 3}) {
					auto const moments = oracle_project(weights, g, count);
					for (Binary solver : {atl::series_power_projection<atl::default_ntt_engine, Vector, Vector>,
										  atl::experimental::series_power_projection_by_padded_bostan_mori<atl::default_ntt_engine, Vector, Vector>,
										  atl::experimental::series_power_projection_by_bostan_mori<atl::default_ntt_engine, Vector, Vector>,
										  atl::experimental::series_power_projection_by_repeated_products<atl::default_ntt_engine, Vector, Vector>}) {
						REQUIRE(solver(weights, g, count) == moments);
						REQUIRE(solver(Vector{}, g, count) == Vector(count));
					}
				}
			}
			REQUIRE(inner.size() == n + 2);
			Vector g = inner;
			g[0] = Coefficient(0);
			for (std::size_t cutoff : {1, 4, 8, 16, 32}) {
				REQUIRE(atl::experimental::series_composition_by_cutoffs(outer, g, n, cutoff, cutoff) == oracle_compose(outer, g, n));
			}
		}
	}

	/** @brief Inverse identities checked with independent schoolbook composition in both orders. */
	template<atl::static_prime_mod_int Coefficient>
	void check_inverse() {
		using Vector = std::vector<Coefficient>;
		using Unary = Vector (*)(Vector const &, std::size_t);
		std::mt19937_64 random(91);
		for (std::size_t n : {0, 1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 31, 32, 33, 63, 64, 65}) {
			CAPTURE(n, Coefficient::mod());
			Vector f(n + 2), identity(n);
			for (auto &value : f) value = Coefficient(random());
			f[0] = Coefficient(0);
			f[1] = Coefficient(1 + random() % (Coefficient::mod() - 1));
			if (n >= 2) identity[1] = Coefficient(1);
			auto const inverse = atl::series_compositional_inverse(f, n);
			for (Unary solver : {atl::series_compositional_inverse<atl::default_ntt_engine, Vector>,
								 atl::experimental::series_compositional_inverse_by_composition_newton<atl::default_ntt_engine, Vector>,
								 atl::experimental::series_compositional_inverse_by_coefficient_recurrence<atl::default_ntt_engine, Vector>}) {
				auto const result = solver(f, n);
				REQUIRE(result == inverse);
				Vector prefix(f.begin(), f.begin() + n);
				REQUIRE(oracle_compose(prefix, result, n) == identity);
				REQUIRE(oracle_compose(result, prefix, n) == identity);
			}
			if (n <= Coefficient::mod()) {
				REQUIRE(atl::experimental::series_compositional_inverse_by_baby_giant_lagrange(f, n) == inverse);
				REQUIRE(atl::experimental::series_compositional_inverse_by_projection_lagrange(f, n) == inverse);
			}
		}
	}
}

TEST_CASE("series_composition: all contenders and independent oracles", "[series_composition]") {
	check_family<atl::mod_int32<998244353>>();
	check_family<atl::mod_int32<1000000007>>();
	check_family<atl::mod_int32<7>>();
	check_family<atl::mod_int32<13>>();
	check_family<atl::mod_int32<2>>();
}

TEST_CASE("series_composition: inverse identities including precision beyond the characteristic", "[series_composition]") {
	check_inverse<atl::mod_int32<998244353>>();
	check_inverse<atl::mod_int32<1000000007>>();
	check_inverse<atl::mod_int32<7>>();
	check_inverse<atl::mod_int32<13>>();
	check_inverse<atl::mod_int32<2>>();
}

TEST_CASE("series_composition: finite outer and ignored tails, single-pass ranges", "[series_composition]") {
	using C = atl::mod_int32<7>;
	using V = std::vector<C>;
	V const f{1, 2, 3, 4, 5, 6, 1, 2, 3}, g{2, 3, 4, 5}, w{3, 2, 1};
	auto const f_copy = f, g_copy = g, w_copy = w;
	std::forward_list<C> const outer(f.begin(), f.end()), inner(g.begin(), g.end()), weights(w.begin(), w.end());
	REQUIRE(atl::series_composition(outer, inner, 3) == oracle_compose(f, g, 3));
	REQUIRE(atl::series_power_projection(weights, inner, 11) == oracle_project(w, g, 11));
	REQUIRE(atl::series_composition(V{1, 2}, V{}, 5) == V{1, 0, 0, 0, 0});
	REQUIRE(atl::series_compositional_inverse(V{}, 0) == V{});
	REQUIRE(atl::series_compositional_inverse(V{}, 1) == V{0});
	REQUIRE(atl::series_compositional_inverse(V{0, 0}, 1) == V{0});
	REQUIRE(f == f_copy);
	REQUIRE(g == g_copy);
	REQUIRE(w == w_copy);
}

TEST_CASE("series_composition: large dense identities and transposition", "[series_composition]") {
	using C = atl::mod_int32<998244353>;
	using V = std::vector<C>;
	std::mt19937_64 random(1234);
	// 16387 * 16387 exceeds the internal work limit: the work is linear in the precision, not in outer length times precision.
	for (std::size_t n : {257, 1025, 16387}) {
		CAPTURE(n);
		V f(n), weights(n), outer(n), identity(n);
		for (auto &value : f) value = C(random());
		for (auto &value : weights) value = C(random());
		for (auto &value : outer) value = C(random());
		f[0] = C(0);
		f[1] = C(3);
		identity[1] = C(1);
		auto const inverse = atl::series_compositional_inverse(f, n);
		REQUIRE(atl::series_composition(f, inverse, n) == identity);
		REQUIRE(atl::series_composition(inverse, f, n) == identity);
		REQUIRE(atl::experimental::series_compositional_inverse_by_composition_newton(f, n) == inverse);
		auto const values = atl::series_composition(outer, f, n);
		auto const moments = atl::series_power_projection(weights, f, n);
		REQUIRE(atl::experimental::series_power_projection_by_padded_bostan_mori(weights, f, n) == moments);
		C left(0), right(0);
		for (std::size_t i = 0; i < n; ++i) {
			left += weights[i] * values[i];
			right += outer[i] * moments[i];
		}
		REQUIRE(left == right);
		// The independent oracle has only two outer terms even at the largest precision.
		V const small_outer{2, 5};
		REQUIRE(atl::series_composition(small_outer, inverse, n) == oracle_compose(small_outer, inverse, n));
	}
}

TEST_CASE("series_composition: public and contender contracts", "[series_composition]") {
	using C = atl::mod_int32<7>;
	using V = std::vector<C>;
	using Binary = V (*)(V const &, V const &, std::size_t);
	using Unary = V (*)(V const &, std::size_t);
	V const oversized(atl::maximum_series_length + 1);
	for (Binary solver : {atl::series_composition<atl::default_ntt_engine, V, V>,
						  atl::experimental::series_composition_by_brent_kung<atl::default_ntt_engine, V, V>,
						  atl::experimental::series_composition_by_fast_horner<atl::default_ntt_engine, V, V>,
						  atl::experimental::series_composition_by_schoolbook_horner<atl::default_ntt_engine, V, V>}) {
		CHECK(atl_test::aborts([&] { (void)solver(V{}, V{}, atl::maximum_series_length + 1); }));
		CHECK(atl_test::aborts([&] { (void)solver(oversized, V{}, 0); }));
		REQUIRE(solver(V{1}, oversized, 1) == V{1});
	}
	for (Binary solver : {atl::series_power_projection<atl::default_ntt_engine, V, V>,
						  atl::experimental::series_power_projection_by_padded_bostan_mori<atl::default_ntt_engine, V, V>,
						  atl::experimental::series_power_projection_by_repeated_products<atl::default_ntt_engine, V, V>,
						  atl::experimental::series_power_projection_by_bostan_mori<atl::default_ntt_engine, V, V>}) {
		CHECK(atl_test::aborts([&] { (void)solver(V{}, V{}, atl::maximum_series_length + 1); }));
		CHECK(atl_test::aborts([&] { (void)solver(oversized, V{}, 0); }));
		REQUIRE(solver(V{1}, oversized, 1) == V{1});
	}
	for (Unary solver : {atl::series_compositional_inverse<atl::default_ntt_engine, V>,
						 atl::experimental::series_compositional_inverse_by_composition_newton<atl::default_ntt_engine, V>,
						 atl::experimental::series_compositional_inverse_by_baby_giant_lagrange<atl::default_ntt_engine, V>,
						 atl::experimental::series_compositional_inverse_by_coefficient_recurrence<atl::default_ntt_engine, V>,
						 atl::experimental::series_compositional_inverse_by_projection_lagrange<atl::default_ntt_engine, V>}) {
		CHECK(atl_test::aborts([&] { (void)solver(V{}, atl::maximum_series_length + 1); }));
		CHECK(atl_test::aborts([&] { (void)solver(V{1, 1}, 1); }));
		CHECK(atl_test::aborts([&] { (void)solver(V{1, 1}, 2); }));
		CHECK(atl_test::aborts([&] { (void)solver(V{}, 2); }));
		CHECK(atl_test::aborts([&] { (void)solver(V{0, 0}, 2); }));
		REQUIRE(solver(V{0, 0}, 1) == V{0});
		REQUIRE(solver(V{1}, 0).empty());
	}
	CHECK(atl_test::aborts([] { (void)atl::experimental::series_compositional_inverse_by_projection_lagrange(V{0, 1}, 8); }));
	CHECK(atl_test::aborts([] { (void)atl::experimental::series_compositional_inverse_by_baby_giant_lagrange(V{0, 1}, 8); }));
	for (std::size_t invalid : {std::size_t{0}, atl::maximum_series_length}) {
		CHECK(atl_test::aborts([&] { (void)atl::experimental::series_composition_by_cutoffs(V{}, V{}, 0, invalid, 1); }));
		CHECK(atl_test::aborts([&] { (void)atl::experimental::series_composition_by_cutoffs(V{}, V{}, 0, 1, invalid); }));
	}
}

TEST_CASE("series_composition: detail guards", "[series_composition]") {
	using C = atl::mod_int32<7>;
	using V = std::vector<C>;
	using E = atl::default_ntt_engine;
	using namespace atl::detail;
	CHECK(atl_test::aborts([] { (void)composition_product<E, C>(V{}, V{}, composition_work_limit + 1); }));
	CHECK(atl_test::aborts([] { (void)composition_pack<C>(V{}, 0); }));
	CHECK(atl_test::aborts([] { (void)composition_pack<C>(V{1}, 2); }));
	CHECK(atl_test::aborts([] { (void)composition_pack<C>(V{}, atl::maximum_series_length + 1); }));
	CHECK(atl_test::aborts([] { (void)composition_extract<C>(V{1}, 0, 1, 0); }));
	CHECK(atl_test::aborts([] { (void)composition_extract<C>(V{1}, 1, 0, 0); }));
	CHECK(atl_test::aborts([] { (void)composition_extract<C>(V{1}, 1, 1, 2); }));
	CHECK(atl_test::aborts([] { (void)composition_extract<C>(V{}, 1, 1, 0); }));
	CHECK(atl_test::aborts([] { (void)composition_extract<C>(V{}, 1, composition_work_limit, 0); }));
	CHECK(atl_test::aborts([] { (void)composition_graeffe<E, true, C>(V{}, V{1}, 1, 1); }));
	CHECK(atl_test::aborts([] { (void)composition_graeffe<E, true, C>(V{}, V{1, 0}, 2, 0); }));
	CHECK(atl_test::aborts([] { (void)composition_graeffe<E, true, C>(V{}, V{}, 2, 1); }));
	CHECK(atl_test::aborts([] { (void)composition_graeffe<E, true, C>(V{1}, V{1, 0}, 2, 1); }));
	CHECK(atl_test::aborts([] { (void)composition_graeffe<E, true, C>(V{1, 0, 0, 0}, V{1, 0}, 2, 1); }));
	CHECK(atl_test::aborts([] { (void)composition_denominator<C>(V{}, 1); }));
	CHECK(atl_test::aborts([] { (void)composition_denominator<C>(V{1}, 0); }));
	CHECK(atl_test::aborts([] { (void)composition_project<E, true, C>(V{}, V{1}, 0); }));
	CHECK(atl_test::aborts([] { (void)composition_project_repeated<E, C>(V{}, V{1}, 0); }));
	CHECK(atl_test::aborts([] { (void)composition_transpose<E, C>(V{}, V{1}, 1, 1); }));
	CHECK(atl_test::aborts([] { (void)composition_transpose<E, C>(V{1}, V{1}, 0, 1); }));
	CHECK(atl_test::aborts([] { (void)composition_transpose<E, C>(V{1}, V{1}, 1, 0); }));
	CHECK(atl_test::aborts([] { (void)composition_transpose<E, C>(V{1}, V{}, 1, 1); }));
	CHECK(atl_test::aborts([] { (void)composition_kinoshita_li<E, C>(V{}, V{0}); }));
	CHECK(atl_test::aborts([] { (void)composition_kinoshita_li<E, C>(V{1}, V{1}); }));
	CHECK(atl_test::aborts([] { (void)composition_brent_kung<E, C>(V{}, V{0}); }));
	CHECK(atl_test::aborts([] { (void)composition_brent_kung<E, C>(V{1}, V{1}); }));
	CHECK(atl_test::aborts([] { (void)composition_horner<E, true, C>(V{}, V{}, 1); }));
	CHECK(atl_test::aborts([] { (void)composition_horner<E, false, C>(V{}, V{1}, 1); }));
	CHECK(atl_test::aborts([] { (void)composition_block_size(atl::maximum_series_length + 1); }));
	CHECK(atl_test::aborts([] { (void)composition_dispatch<E, composition_algorithm::kinoshita_li, C>(V{}, V{}, 1); }));
	using Friendly = atl::mod_int32<998244353>;
	CHECK(atl_test::aborts([] { (void)composition_transform<E, Friendly>(std::vector<Friendly>{}, 0); }));
	CHECK(atl_test::aborts([] { (void)composition_transform<E, Friendly>(std::vector<Friendly>{}, 3); }));
	CHECK(atl_test::aborts([] { (void)composition_transform<E, Friendly>(std::vector<Friendly>{1, 2}, 1); }));
}

TEST_CASE("series_composition: half-transform schedule and denominator reuse", "[series_composition]") {
	using C = atl::mod_int32<998244353>;
	using V = std::vector<C>;
	using namespace atl::detail;
	V series(64, C(3)), numerator(64, C(5));
	series[0] = C(0);
	auto const denominator = composition_denominator<C>(series, 64);
	auto const expected = composition_graeffe<atl::default_ntt_engine, false, C>(numerator, denominator, 64, 64);
	counting_engine::forwards.clear();
	counting_engine::inverses.clear();
	auto const actual = composition_graeffe<counting_engine, true, C>(numerator, denominator, 64, 64);
	REQUIRE(actual.numerator == expected.numerator);
	REQUIRE(actual.denominator == expected.denominator);
	REQUIRE(actual.spectrum.size() == 512);
	REQUIRE(counting_engine::forwards == std::vector<std::size_t>{256, 256, 256, 256});
	REQUIRE(counting_engine::inverses == std::vector<std::size_t>{256, 256});
	counting_engine::forwards.clear();
	counting_engine::inverses.clear();
	auto const prepared = composition_graeffe<counting_engine, true, C>({}, denominator, 64, 64);
	REQUIRE(prepared.denominator == expected.denominator);
	REQUIRE(counting_engine::forwards == std::vector<std::size_t>{256, 256});
	REQUIRE(counting_engine::inverses == std::vector<std::size_t>{256});
	counting_engine::forwards.clear();
	counting_engine::inverses.clear();
	REQUIRE(atl::series_composition<unsupported_engine>(numerator, series, 64) == oracle_compose(numerator, series, 64));
	REQUIRE(atl::series_power_projection<unsupported_engine>(numerator, series, 64) == oracle_project(numerator, series, 64));
	REQUIRE(counting_engine::forwards.empty());
	REQUIRE(counting_engine::inverses.empty());
	// With one nonzero y-row the transpose is exactly multiplication by 1-yG; its cached forward is reused on ascent.
	V const outer{1, 2};
	counting_engine::forwards.clear();
	counting_engine::inverses.clear();
	auto const transposed = composition_kinoshita_li<counting_engine, C>(outer, series);
	REQUIRE(transposed == oracle_compose(outer, series, 64));
	REQUIRE(counting_engine::forwards.size() == 18);
	REQUIRE(counting_engine::inverses.size() == 12);
}

TEST_CASE("series_composition: wide prime products and large nonzero constants", "[series_composition]") {
	using C = atl::mod_int64<18446744073709551557ULL>;
	using V = std::vector<C>;
	V f(25), g(27), w(25);
	for (std::size_t i = 0; i < f.size(); ++i) {
		f[i] = C(-1) - C(i);
		g[i] = C(i + 3);
		w[i] = C(i * i + 1);
	}
	REQUIRE(atl::series_composition(f, g, 25) == oracle_compose(f, g, 25));
	REQUIRE(atl::series_power_projection(w, g, 29) == oracle_project(w, g, 29));
	f[0] = C(0);
	f[1] = C(3);
	auto const inverse = atl::series_compositional_inverse(f, 25);
	REQUIRE(oracle_compose(f, inverse, 25) == V{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
	using Friendly = atl::mod_int32<998244353>;
	std::vector<Friendly> outer(257), inner(129), weights(129);
	for (std::size_t i = 0; i < outer.size(); ++i) outer[i] = Friendly(i + 7);
	for (std::size_t i = 0; i < inner.size(); ++i) inner[i] = Friendly(i + 1);
	for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = Friendly(i * 3);
	REQUIRE(atl::series_composition(outer, inner, 129) == oracle_compose(outer, inner, 129));
	REQUIRE(atl::series_power_projection(weights, inner, 193) == oracle_project(weights, inner, 193));
}

TEST_CASE("series_composition: blocked packed products", "[series_composition]") {
	using C = atl::mod_int32<998244353>;
	using V = std::vector<C>;
	V left(47), right(39);
	for (std::size_t i = 0; i < left.size(); ++i) left[i] = C(i * i + 3);
	for (std::size_t i = 0; i < right.size(); ++i) right[i] = C(i * 5 + 2);
	for (std::size_t block : {1, 2, 8, 16, 32}) {
		for (std::size_t length : {0, 1, 2, 31, 64, 85, 93})
			REQUIRE(atl::detail::composition_product<atl::default_ntt_engine, C>(left, right, length, block) == oracle_multiply(left, right, length));
	}
	CHECK(atl_test::aborts([&] { (void)atl::detail::composition_product<atl::default_ntt_engine, C>(left, right, 1, 0); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::composition_product<atl::default_ntt_engine, C>(left, right, 1, atl::maximum_series_length); }));
}

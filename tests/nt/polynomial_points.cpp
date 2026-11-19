#include <atl/experimental/nt/polynomial_points_variants.hpp>
#include <atl/nt/detail/polynomial_points.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/ntt_engine.hpp>
#include <atl/nt/polynomial_points.hpp>
#include <atl/nt/prime_mod_int.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <forward_list>
#include <random>
#include <span>
#include <vector>

#include "../expect_abort.hpp"

namespace {
	/** @brief Independent scalar oracle for every evaluation-like operation. */
	template<atl::static_prime_mod_int Coefficient>
	Coefficient oracle_evaluate(std::vector<Coefficient> const &polynomial, Coefficient point) {
		Coefficient result(0);
		for (auto iterator = polynomial.rbegin(); iterator != polynomial.rend(); ++iterator) result = result * point + *iterator;
		return result;
	}

	/** @brief Independent full schoolbook product, with its prescribed length. */
	template<atl::static_prime_mod_int Coefficient>
	std::vector<Coefficient> oracle_multiply(std::vector<Coefficient> const &left, std::vector<Coefficient> const &right) {
		if (left.empty() || right.empty()) return {};
		std::vector<Coefficient> result(left.size() + right.size() - 1);
		for (std::size_t i = 0; i < left.size(); ++i)
			for (std::size_t j = 0; j < right.size(); ++j) result[i + j] += left[i] * right[j];
		return result;
	}

	/** @brief Binomial expansion using Pascal rows, including lengths beyond the characteristic. */
	template<atl::static_prime_mod_int Coefficient>
	std::vector<Coefficient> shift_oracle(std::vector<Coefficient> const &polynomial, Coefficient shift) {
		std::vector<Coefficient> result(polynomial.size()), binomial{Coefficient(1)};
		for (std::size_t degree = 0; degree < polynomial.size(); ++degree) {
			Coefficient power(1);
			for (std::size_t j = degree + 1; j > 0; --j) {
				result[j - 1] += polynomial[degree] * binomial[j - 1] * power;
				power *= shift;
			}
			binomial.push_back(Coefficient(1));
			for (std::size_t j = degree; j > 0; --j) binomial[j] += binomial[j - 1];
		}
		return result;
	}

	/** @brief Schoolbook Lagrange interpolation, independent of all tree and difference kernels. */
	template<atl::static_prime_mod_int Coefficient>
	std::vector<Coefficient> interpolate_oracle(std::vector<Coefficient> const &points, std::vector<Coefficient> const &values) {
		std::vector<Coefficient> result(points.size());
		for (std::size_t i = 0; i < points.size(); ++i) {
			std::vector<Coefficient> basis{Coefficient(1)};
			Coefficient denominator(1);
			for (std::size_t j = 0; j < points.size(); ++j) {
				if (i == j) continue;
				basis = oracle_multiply(basis, std::vector<Coefficient>{-points[j], Coefficient(1)});
				denominator *= points[i] - points[j];
			}
			Coefficient const scale = values[i] / denominator;
			for (std::size_t j = 0; j < basis.size(); ++j) result[j] += scale * basis[j];
		}
		return result;
	}

	/** @brief Exercise all contenders around the tree, schoolbook, and Newton cutoffs. */
	template<atl::static_prime_mod_int Coefficient>
	void check_field() {
		using Vector = std::vector<Coefficient>;
		std::mt19937_64 random(7001);
		for (std::size_t n : {0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 16, 17, 23, 24, 25, 31, 32, 33, 47, 48, 49, 63, 64, 65, 95, 96, 97, 127, 128, 129, 255, 256, 257}) {
			CAPTURE(Coefficient::mod(), n);
			Vector polynomial(n), nodes(n + 3);
			for (auto &value : polynomial) value = Coefficient(random());
			for (auto &point : nodes) point = Coefficient(random());
			if (!polynomial.empty()) polynomial.back() = Coefficient(0);
			Coefficient const shift(random());
			auto const shifted = shift_oracle(polynomial, shift);
			REQUIRE(atl::polynomial_taylor_shift(polynomial, shift) == shifted);
			REQUIRE(atl::experimental::polynomial_taylor_shift_by_divide_and_conquer(polynomial, shift) == shifted);
			REQUIRE(atl::experimental::polynomial_taylor_shift_by_horner(polynomial, shift) == shifted);
			Vector roots(n);
			for (auto &root : roots) root = Coefficient(random());
			Vector product{Coefficient(1)};
			for (auto root : roots) product = oracle_multiply(product, Vector{-root, Coefficient(1)});
			REQUIRE(atl::polynomial_from_roots(roots) == product);
			REQUIRE(atl::experimental::polynomial_from_roots_by_sequential_product(roots) == product);
			for (std::size_t length : {n / 2, n, 2 * n + 1}) {
				Vector input(length);
				for (auto &value : input) value = Coefficient(random());
				Vector expected;
				for (auto point : nodes) expected.push_back(oracle_evaluate(input, point));
				REQUIRE(atl::polynomial_evaluation(input, nodes) == expected);
				REQUIRE(atl::experimental::polynomial_evaluation_by_remainder_tree(input, nodes) == expected);
				REQUIRE(atl::experimental::polynomial_evaluation_by_horner(input, nodes) == expected);
				for (std::size_t cutoff : {16, 24, 32, 48, 64, 96, 128})
					REQUIRE(atl::experimental::polynomial_evaluation_by_leaf_cutoff(input, nodes, cutoff) == expected);
			}
			for (Coefficient first : {Coefficient(0), Coefficient(1), Coefficient(3)}) {
				for (Coefficient ratio : {Coefficient(0), Coefficient(1), Coefficient(-1), Coefficient(3)}) {
					Vector expected(nodes.size());
					Coefficient point = first;
					for (auto &value : expected) {
						value = oracle_evaluate(polynomial, point);
						point *= ratio;
					}
					REQUIRE(atl::polynomial_geometric_evaluation(polynomial, first, ratio, nodes.size()) == expected);
					REQUIRE(atl::experimental::polynomial_geometric_evaluation_by_horner(polynomial, first, ratio, nodes.size()) == expected);
				}
			}
			if (n > Coefficient::mod()) continue;
			Vector points(n), values(n), samples(n);
			for (std::size_t i = 0; i < n; ++i) {
				points[i] = Coefficient(3 * i + 1);
				values[i] = Coefficient(random());
				samples[i] = oracle_evaluate(polynomial, Coefficient(i));
			}
			auto const interpolated = interpolate_oracle(points, values);
			REQUIRE(atl::polynomial_interpolation(points, values) == interpolated);
			REQUIRE(atl::experimental::polynomial_interpolation_by_remainder_tree(points, values) == interpolated);
			REQUIRE(atl::experimental::polynomial_interpolation_by_newton_differences(points, values) == interpolated);
			for (Coefficient target : {Coefficient(0), Coefficient(1), Coefficient(-2), shift}) {
				std::size_t const count = n + 17;
				Vector expected(count);
				for (std::size_t j = 0; j < count; ++j) expected[j] = oracle_evaluate(polynomial, target + Coefficient(j));
				REQUIRE(atl::sample_point_shift(samples, target, count) == expected);
				REQUIRE(atl::experimental::sample_point_shift_by_falling_factorials(samples, target, count) == expected);
				REQUIRE(atl::experimental::sample_point_shift_by_direct_barycentric(samples, target, count) == expected);
			}
		}
	}

	/** @brief Scalar engine that counts its transform calls; a fused convolution counts as two forwards and one inverse. */
	struct counting_engine {
		static inline std::size_t forwards = 0;
		static inline std::size_t inverses = 0;
		static inline atl::uint32 maximum_height = 0;

		template<atl::static_mod_int Coefficient>
		static consteval bool supports() { return atl::scalar_ntt::supports<Coefficient>(); }

		template<atl::static_mod_int Coefficient>
		static void forward(decltype(Coefficient::mod()) *data, atl::uint32 height) {
			++forwards;
			maximum_height = std::max(maximum_height, height);
			atl::scalar_ntt::forward<Coefficient>(data, height);
		}

		template<atl::static_mod_int Coefficient>
		static void inverse(decltype(Coefficient::mod()) *data, atl::uint32 height) {
			++inverses;
			atl::scalar_ntt::inverse<Coefficient>(data, height);
		}

		template<atl::static_mod_int Coefficient>
		static void convolve(decltype(Coefficient::mod()) *left, decltype(Coefficient::mod()) *right, atl::uint32 height) {
			forwards += 2;
			maximum_height = std::max(maximum_height, height);
			++inverses;
			atl::scalar_ntt::convolve<Coefficient>(left, right, height);
		}
	};
}

TEST_CASE("polynomial_points: every contender versus independent oracles", "[polynomial_points]") {
	check_field<atl::mod_int32<998244353>>();
	check_field<atl::mod_int32<1000000007>>();
	check_field<atl::mod_int32<7>>();
	check_field<atl::mod_int32<13>>();
}

TEST_CASE("polynomial_points: normalized products and input ranges", "[polynomial_points]") {
	using Coefficient = atl::mod_int32<998244353>;
	using Vector = std::vector<Coefficient>;
	std::mt19937_64 random(38);
	for (std::size_t count : {0, 1, 2, 17, 33, 65}) {
		std::vector<Vector> factors;
		Vector expected{1};
		for (std::size_t i = 0; i < count; ++i) {
			Vector factor(1 + i % 9);
			for (auto &value : factor) value = Coefficient(random());
			expected = oracle_multiply(expected, factor);
			factor.resize(factor.size() + 3);
			factors.push_back(factor);
		}
		while (!expected.empty() && expected.back() == Coefficient(0)) expected.pop_back();
		REQUIRE(atl::polynomial_product(factors) == expected);
		REQUIRE(atl::experimental::polynomial_product_by_smallest_first(factors) == expected);
		REQUIRE(atl::experimental::polynomial_product_by_sequential_schoolbook(factors) == expected);
		for (Vector zero : {Vector{}, Vector{0, 0}}) {
			factors.push_back(zero);
			REQUIRE(atl::polynomial_product(factors).empty());
			REQUIRE(atl::experimental::polynomial_product_by_smallest_first(factors).empty());
			REQUIRE(atl::experimental::polynomial_product_by_sequential_schoolbook(factors).empty());
		}
	}
	std::forward_list<Coefficient> const input{1, 2, 0}, points{3, 3, 0};
	REQUIRE(atl::polynomial_taylor_shift(input, Coefficient(1)) == Vector{3, 2, 0});
	REQUIRE(atl::polynomial_evaluation(input, points) == Vector{7, 7, 1});
	REQUIRE(atl::polynomial_interpolation(std::forward_list<Coefficient>{0, 1}, std::forward_list<Coefficient>{1, 3}) == Vector{1, 2});
	REQUIRE(atl::polynomial_product(std::forward_list<std::forward_list<Coefficient>>{{1, 1, 0}, {1, 0}}) == Vector{1, 1});
	REQUIRE(atl::polynomial_evaluation(Vector{1, 2}, Vector{}).empty());
	REQUIRE(atl::polynomial_geometric_evaluation(input, Coefficient(3), Coefficient(2), 0).empty());
	REQUIRE(atl::sample_point_shift(input, Coefficient(3), 0).empty());
}

TEST_CASE("polynomial_points: reused tree and large round trips", "[polynomial_points]") {
	auto const check = []<atl::static_prime_mod_int Coefficient>() {
		using Vector = std::vector<Coefficient>;
		for (std::size_t n : {std::size_t{1023}, std::size_t{1024}, std::size_t{1025}, (std::size_t{1} << 16) + 3}) {
			CAPTURE(n);
			Vector points(n);
			for (std::size_t i = 0; i < n; ++i) points[i] = Coefficient(i + 1);
			atl::polynomial_point_tree<Coefficient> const tree(points);
			REQUIRE(tree.point_count() == n);
			for (std::size_t length : {std::size_t{0}, std::size_t{3}, n}) {
				Vector polynomial(length);
				for (std::size_t i = 0; i < length; ++i) polynomial[i] = Coefficient(i * i + 13);
				auto const values = tree.evaluate(polynomial);
				auto expected = polynomial;
				expected.resize(n);
				REQUIRE(tree.interpolate(values) == expected);
				REQUIRE(tree.evaluate(tree.interpolate(values)) == values);
				for (std::size_t i : {std::size_t{0}, n / 2, n - 1}) REQUIRE(values[i] == oracle_evaluate(polynomial, points[i]));
			}
		}
	};
	check.operator()<atl::mod_int32<998244353>>();
	check.operator()<atl::mod_int32<1000000007>>();
}

TEST_CASE("polynomial_points: cached transform schedule", "[polynomial_points]") {
	using Coefficient = atl::mod_int32<998244353>;
	std::vector<Coefficient> points(128), weights(128, Coefficient(1)), values(128);
	for (std::size_t i = 0; i < points.size(); ++i) points[i] = Coefficient(i);
	atl::detail::points_tree_storage<Coefficient, counting_engine> const tree(points);
	counting_engine::forwards = counting_engine::inverses = 0;
	REQUIRE(tree.evaluate(weights) == atl::polynomial_evaluation(weights, points));
	// Root preparation: inverse (3/2), product (2/1); three internal nodes (1/2 each).
	REQUIRE(counting_engine::forwards == 8);
	REQUIRE(counting_engine::inverses == 9);
	counting_engine::forwards = counting_engine::inverses = 0;
	REQUIRE(tree.interpolate(values) == std::vector<Coefficient>(128));
	// Derivative evaluation as above, then three upward nodes (2/1 each).
	REQUIRE(counting_engine::forwards == 14);
	REQUIRE(counting_engine::inverses == 12);
}

TEST_CASE("polynomial_points: asserted public and contender contracts", "[polynomial_points]") {
	using Coefficient = atl::mod_int32<7>;
	using Vector = std::vector<Coefficient>;
	using Binary = Vector (*)(Vector const &, Vector const &);
	for (Binary solver : {atl::polynomial_interpolation<atl::default_ntt_engine, Vector, Vector>, atl::experimental::polynomial_interpolation_by_remainder_tree<atl::default_ntt_engine, Vector, Vector>, atl::experimental::polynomial_interpolation_by_newton_differences<atl::default_ntt_engine, Vector, Vector>}) {
		CHECK(atl_test::aborts([&] { (void)solver(Vector{1}, Vector{}); }));
		CHECK(atl_test::aborts([&] { (void)solver(Vector{1, 2, 1}, Vector{3, 4, 5}); }));
	}
	using Sample = Vector (*)(Vector const &, Coefficient, std::size_t);
	for (Sample solver : {atl::sample_point_shift<atl::default_ntt_engine, Vector>, atl::experimental::sample_point_shift_by_falling_factorials<atl::default_ntt_engine, Vector>, atl::experimental::sample_point_shift_by_direct_barycentric<atl::default_ntt_engine, Vector>}) {
		CHECK(atl_test::aborts([&] { (void)solver(Vector(8), Coefficient(0), 1); }));
		CHECK(atl_test::aborts([&] { (void)solver(Vector{}, Coefficient(0), atl::maximum_series_length + 1); }));
	}
	Vector oversized(atl::maximum_series_length + 1);
	CHECK(atl_test::aborts([&] { (void)atl::polynomial_taylor_shift(oversized, Coefficient(1)); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::polynomial_taylor_shift_by_divide_and_conquer(oversized, Coefficient(1)); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::polynomial_taylor_shift_by_horner(oversized, Coefficient(1)); }));
	CHECK(atl_test::aborts([&] { (void)atl::polynomial_from_roots(oversized); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::polynomial_from_roots_by_sequential_product(oversized); }));
	CHECK(atl_test::aborts([&] { (void)atl::polynomial_evaluation(oversized, Vector{}); }));
	CHECK(atl_test::aborts([&] { (void)atl::polynomial_evaluation(Vector{}, oversized); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::polynomial_evaluation_by_remainder_tree(oversized, Vector{}); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::polynomial_evaluation_by_horner(Vector{}, oversized); }));
	CHECK(atl_test::aborts([&] { (void)atl::polynomial_geometric_evaluation(Vector{}, Coefficient(1), Coefficient(1), atl::maximum_series_length + 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::polynomial_geometric_evaluation_by_horner(Vector{}, Coefficient(1), Coefficient(1), atl::maximum_series_length + 1); }));
	for (auto solver : {atl::polynomial_product<atl::default_ntt_engine, std::vector<Vector>>, atl::experimental::polynomial_product_by_smallest_first<atl::default_ntt_engine, std::vector<Vector>>, atl::experimental::polynomial_product_by_sequential_schoolbook<atl::default_ntt_engine, std::vector<Vector>>}) {
		CHECK(atl_test::aborts([&] { (void)solver(std::vector<Vector>{oversized}); }));
		CHECK(atl_test::aborts([&] { (void)solver(std::vector<Vector>{Vector(atl::maximum_series_length, Coefficient(1)), Vector{1, 1}}); }));
	}
	CHECK(atl_test::aborts([&] { (void)atl::experimental::polynomial_evaluation_by_leaf_cutoff(Vector{}, Vector{}, 0); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::polynomial_evaluation_by_leaf_cutoff(Vector{}, Vector{}, atl::maximum_series_length); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::points_middle_product<atl::default_ntt_engine, Coefficient>(Vector{}, Vector{}); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::points_middle_product<atl::default_ntt_engine, Coefficient>(Vector{1}, Vector{1, 2}); }));
	CHECK(atl_test::aborts([&] { (void)atl::polynomial_from_roots(Vector(atl::maximum_series_length)); }));
	atl::polynomial_point_tree<Coefficient> const tree(Vector{1});
	CHECK(atl_test::aborts([&] { (void)tree.interpolate(Vector{}); }));
}

TEST_CASE("polynomial_points: internal guards", "[polynomial_points]") {
	using Coefficient = atl::mod_int32<7>;
	using Vector = std::vector<Coefficient>;
	using Engine = atl::default_ntt_engine;
	Vector oversized(atl::maximum_series_length + 1);
	std::span<Coefficient const> const input(oversized);
	CHECK(atl_test::aborts([] { (void)atl::detail::points_can_transform<Coefficient>(0); }));
	CHECK(atl_test::aborts([] { (void)atl::detail::points_can_transform<Coefficient>(3); }));
	CHECK(atl_test::aborts([] { (void)atl::detail::points_can_transform<Coefficient>(2 * atl::maximum_series_length); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::points_middle_product<Engine, Coefficient>(input, Vector{1}); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::points_shift_horner<Coefficient>(input, Coefficient(0)); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::points_shift_divide<Engine, Coefficient>(input, Coefficient(0)); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::points_shift_factorial<Engine, Coefficient>(input, Coefficient(0)); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::points_roots_sequential<Coefficient>(input); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::points_roots_balanced<Engine, Coefficient>(input); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::points_geometric<Engine, Coefficient>(input, Coefficient(0), Coefficient(0), 0); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::points_geometric<Engine, Coefficient>(Vector{}, Coefficient(0), Coefficient(0), atl::maximum_series_length + 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::points_sample_middle<Engine, Coefficient>(Vector(8), Coefficient(0), 0); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::points_sample_middle<Engine, Coefficient>(Vector{}, Coefficient(0), atl::maximum_series_length + 1); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::points_interpolation_newton<Coefficient>(Vector{0}, Vector{}); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::points_interpolation_newton<Coefficient>(input, input); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::points_interpolation_newton<Coefficient>(Vector{0, 0}, Vector{1, 1}); }));
	CHECK(atl_test::aborts([&] { (void)atl::detail::points_tree_storage<Coefficient, Engine>(oversized); }));
	atl::detail::points_tree_storage<Coefficient, Engine> const tree(Vector{0});
	CHECK(atl_test::aborts([&] { (void)tree.evaluate(oversized); }));
	using Large = atl::mod_int32<998244353>;
	CHECK(atl_test::aborts([] { (void)atl::detail::points_sample_middle<Engine, Large>(std::vector<Large>(atl::maximum_series_length + 1), Large(0), 0); }));
}

TEST_CASE("polynomial_points: characteristic two and wide primes", "[polynomial_points]") {
	using Bit = atl::mod_int32<2>;
	using Bits = std::vector<Bit>;
	REQUIRE(atl::polynomial_taylor_shift(Bits{1, 0, 1, 0, 1}, Bit(1)) == Bits{1, 0, 1, 0, 1});
	REQUIRE(atl::polynomial_evaluation(Bits{1, 1, 1}, Bits{0, 1, 0}) == Bits{1, 1, 1});
	REQUIRE(atl::polynomial_interpolation(Bits{0, 1}, Bits{1, 0}) == Bits{1, 1});
	REQUIRE(atl::sample_point_shift(Bits{1, 0}, Bit(1), 7) == Bits{0, 1, 0, 1, 0, 1, 0});
	REQUIRE(atl::experimental::sample_point_shift_by_falling_factorials(Bits{1, 0}, Bit(1), 7) == Bits{0, 1, 0, 1, 0, 1, 0});
	using Wide = atl::mod_int64<18446744073709551557ULL>;
	using Vector = std::vector<Wide>;
	Vector polynomial(65), points(67), expected;
	for (std::size_t i = 0; i < polynomial.size(); ++i) polynomial[i] = Wide(-1) - Wide(i);
	for (std::size_t i = 0; i < points.size(); ++i) points[i] = Wide(i);
	for (auto point : points) expected.push_back(oracle_evaluate(polynomial, point));
	REQUIRE(atl::polynomial_evaluation(polynomial, points) == expected);
	polynomial.resize(points.size());
	REQUIRE(atl::polynomial_interpolation(points, expected) == polynomial);
	REQUIRE(atl::polynomial_taylor_shift(polynomial, Wide(3)) == shift_oracle(polynomial, Wide(3)));
	Vector shifted_samples(70), geometric(70);
	Wide point(3);
	for (std::size_t j = 0; j < 70; ++j) {
		shifted_samples[j] = oracle_evaluate(polynomial, Wide(-2) + Wide(j));
		geometric[j] = oracle_evaluate(polynomial, point);
		point *= Wide(-1);
	}
	REQUIRE(atl::sample_point_shift(expected, Wide(-2), 70) == shifted_samples);
	REQUIRE(atl::polynomial_geometric_evaluation(polynomial, Wide(3), Wide(-1), 70) == geometric);
}

TEST_CASE("polynomial_points: duplicate nodes across tree branches", "[polynomial_points]") {
	using Coefficient = atl::mod_int32<998244353>;
	using Vector = std::vector<Coefficient>;
	Vector points(65), values(65);
	for (std::size_t i = 0; i < points.size(); ++i) points[i] = Coefficient(i);
	points.back() = points.front();
	Vector const polynomial{1, 3, 4};
	for (std::size_t i = 0; i < points.size(); ++i) values[i] = oracle_evaluate(polynomial, points[i]);
	REQUIRE(atl::polynomial_evaluation(polynomial, points) == values);
	CHECK(atl_test::aborts([&] { (void)atl::polynomial_interpolation(points, values); }));
	CHECK(atl_test::aborts([&] { (void)atl::experimental::polynomial_interpolation_by_remainder_tree(points, values); }));
}

TEST_CASE("polynomial_points: middle products keep the smaller transform", "[polynomial_points]") {
	using Coefficient = atl::mod_int32<998244353>;
	std::vector<Coefficient> polynomial(256, Coefficient(1));
	counting_engine::maximum_height = 0;
	counting_engine::forwards = counting_engine::inverses = 0;
	REQUIRE(atl::polynomial_geometric_evaluation<counting_engine>(polynomial, Coefficient(3), Coefficient(5), 256)
			== atl::experimental::polynomial_geometric_evaluation_by_horner(polynomial, Coefficient(3), Coefficient(5), 256));
	REQUIRE(counting_engine::maximum_height == 9);
	REQUIRE(counting_engine::forwards == 2);
	REQUIRE(counting_engine::inverses == 1);
	counting_engine::maximum_height = 0;
	counting_engine::forwards = counting_engine::inverses = 0;
	REQUIRE(atl::sample_point_shift<counting_engine>(polynomial, Coefficient(700), 256) == polynomial);
	REQUIRE(counting_engine::maximum_height == 9);
	REQUIRE(counting_engine::forwards == 2);
	REQUIRE(counting_engine::inverses == 1);
}

TEST_CASE("polynomial_points: blocked middle products", "[polynomial_points]") {
	auto const check = []<atl::static_prime_mod_int Coefficient>() {
		using Vector = std::vector<Coefficient>;
		using Engine = atl::default_ntt_engine;
		for (std::size_t n : {1, 2, 6, 32, 33, 63, 64, 65, 129}) {
			Vector polynomial(n), samples(n);
			for (std::size_t i = 0; i < n; ++i) polynomial[i] = Coefficient(i * i + 1);
			for (std::size_t i = 0; i < n; ++i) samples[i] = oracle_evaluate(polynomial, Coefficient(i));
			for (std::size_t count : {1, 33, 65, 131}) {
				for (std::size_t limit : {2, 16, 64}) {
					Vector expected(count);
					for (std::size_t j = 0; j < count; ++j) expected[j] = oracle_evaluate(polynomial, Coefficient(3) * Coefficient(5).power(j));
					REQUIRE(atl::detail::points_geometric<Engine, Coefficient>(polynomial, Coefficient(3), Coefficient(5), count, limit) == expected);
					if (n > Coefficient::mod()) continue;
					for (std::size_t j = 0; j < count; ++j) expected[j] = oracle_evaluate(polynomial, Coefficient(-2) + Coefficient(j));
					REQUIRE(atl::detail::points_sample_middle<Engine, Coefficient>(samples, Coefficient(-2), count, limit) == expected);
				}
			}
		}
	};
	check.operator()<atl::mod_int32<998244353>>();
	check.operator()<atl::mod_int32<1000000007>>();
	check.operator()<atl::mod_int32<13>>();
	using Coefficient = atl::mod_int32<7>;
	using Vector = std::vector<Coefficient>;
	using Engine = atl::default_ntt_engine;
	for (std::size_t limit : {std::size_t{0}, std::size_t{1}, atl::maximum_series_length + 1}) {
		CHECK(atl_test::aborts([&] { (void)atl::detail::points_geometric<Engine, Coefficient>(Vector{}, Coefficient(0), Coefficient(0), 0, limit); }));
		CHECK(atl_test::aborts([&] { (void)atl::detail::points_sample_middle<Engine, Coefficient>(Vector{}, Coefficient(0), 0, limit); }));
	}
}

TEST_CASE("polynomial_points: tree points are the constructor's points", "[polynomial_points]") {
	using Coefficient = atl::mod_int32<998244353>;
	using Vector = std::vector<Coefficient>;
	for (std::size_t n : {0, 1, 5, 32, 33, 100}) {
		CAPTURE(n);
		Vector points(n);
		for (std::size_t i = 0; i < n; ++i) points[i] = Coefficient((n - i) % 7);
		std::forward_list<Coefficient> const single_pass(points.begin(), points.end());
		atl::polynomial_point_tree<Coefficient> const tree(single_pass);
		std::span<Coefficient const> const view = tree.points();
		REQUIRE(view.size() == n);
		REQUIRE(view.size() == tree.point_count());
		REQUIRE(std::ranges::equal(view, points));
	}
}

TEST_CASE("polynomial_points: Taylor shift on both sides of n = p", "[polynomial_points]") {
	auto const check = []<atl::static_prime_mod_int Coefficient>() {
		using Vector = std::vector<Coefficient>;
		std::size_t const p = std::size_t(Coefficient::mod());
		std::mt19937_64 random(97);
		for (std::size_t n : {p - 1, p, p + 1, 4 * p + 1}) {
			for (bool trailing_zero : {false, true}) {
				CAPTURE(Coefficient::mod(), n, trailing_zero);
				Vector polynomial(n);
				for (auto &value : polynomial) value = Coefficient(random());
				if (trailing_zero) polynomial.back() = Coefficient(0);
				Coefficient const shift(random() % p + 1);
				REQUIRE(atl::polynomial_taylor_shift(polynomial, shift) == shift_oracle(polynomial, shift));
			}
		}
	};
	check.operator()<atl::mod_int32<7>>();
	check.operator()<atl::mod_int32<11>>();
}

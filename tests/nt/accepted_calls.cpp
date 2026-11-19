#include <atl/experimental/nt/bitwise_convolution_variants.hpp>
#include <atl/experimental/nt/convolution_variants.hpp>
#include <atl/experimental/nt/factorization_variants.hpp>
#include <atl/experimental/nt/online_convolution_variants.hpp>
#include <atl/experimental/nt/polynomial_gcd_variants.hpp>
#include <atl/experimental/nt/polynomial_points_variants.hpp>
#include <atl/experimental/nt/power_series_variants.hpp>
#include <atl/experimental/nt/series_composition_variants.hpp>
#include <atl/experimental/nt/sieve_variants.hpp>
#include <atl/experimental/nt/square_root_mod_variants.hpp>
#include <atl/nt/batch_inverse.hpp>
#include <atl/nt/bitwise_convolution.hpp>
#include <atl/nt/convolution.hpp>
#include <atl/nt/crt.hpp>
#include <atl/nt/discrete_log.hpp>
#include <atl/nt/dynamic_mod_int.hpp>
#include <atl/nt/factorial_table.hpp>
#include <atl/nt/fft.hpp>
#include <atl/nt/floor_sum.hpp>
#include <atl/nt/gcd.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/modular.hpp>
#include <atl/nt/montgomery.hpp>
#include <atl/nt/multiplicative.hpp>
#include <atl/nt/online_convolution.hpp>
#include <atl/nt/polynomial_gcd.hpp>
#include <atl/nt/polynomial_points.hpp>
#include <atl/nt/power.hpp>
#include <atl/nt/power_series.hpp>
#include <atl/nt/series_calculus.hpp>
#include <atl/nt/series_composition.hpp>
#include <atl/nt/square_root_mod.hpp>
#include <atl/nt/static_modulus.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <functional>
#include <list>
#include <ranges>
#include <span>
#include <sstream>
#include <utility>
#include <vector>
#include "input_only.hpp"

namespace {
	struct explicit_identity {
		int value;

		explicit constexpr explicit_identity(int v) : value(v) {}

		friend constexpr explicit_identity operator*(explicit_identity const &a, explicit_identity const &b) {
			return explicit_identity(a.value * b.value);
		}
	};

	struct void_assignment {
		int value;

		explicit constexpr void_assignment(int v) : value(v) {}

		constexpr void_assignment(void_assignment const &) = default;

		constexpr void operator=(void_assignment const &rhs) { value = rhs.value; }

		friend constexpr void_assignment operator*(void_assignment const &a, void_assignment const &b) {
			return void_assignment(a.value * b.value);
		}
	};

	struct stateful_identity {
		int value;
		int context;

		friend constexpr stateful_identity operator*(stateful_identity a, stateful_identity b) {
			return {a.value * b.value, a.context + b.context};
		}
	};

	struct move_only_base {
		int value;

		explicit constexpr move_only_base(int v) : value(v) {}

		constexpr move_only_base(move_only_base &&) = default;
		constexpr move_only_base &operator=(move_only_base &&) = default;

		friend constexpr move_only_base operator*(move_only_base const &a, move_only_base const &b) {
			return move_only_base(a.value * b.value);
		}
	};

	static_assert(atl::power(explicit_identity(3), 3u).value == 27);
	static_assert(atl::power(void_assignment(3), 3u).value == 27);
	static_assert(atl::power(stateful_identity{3, 1}, 3u, stateful_identity{1, 0}).context == 3);
	static_assert(atl::power(move_only_base(3), 3u).value == 27);
	static_assert(atl::power(2u, atl::uint128(10)) == 1024);
	static_assert(std::same_as<atl::modulus_engine<atl::uint64(5)>, atl::montgomery<atl::uint64(5)>>);
	static_assert(std::same_as<atl::modulus_engine<atl::uint64(4)>, atl::static_modulus<atl::uint64(4)>>);
	static_assert(std::same_as<atl::modulus_engine<(atl::uint128(1) << 89) - 1>,
							   atl::static_modulus<(atl::uint128(1) << 89) - 1>>);
	static_assert(atl::montgomery<atl::uint8(251)>::from_form(atl::montgomery<atl::uint8(251)>::to_form(250)) == 250);
	static_assert(atl::extended_gcd(atl::uint128(12), atl::uint128(18)).gcd == 6);
	static_assert(atl::extended_gcd(atl::int128(-12), atl::int128(18)).gcd == 6);
	static_assert(atl::mod_int<atl::uint128, (atl::uint128(1) << 89) - 1>(-1).value() == (atl::uint128(1) << 89) - 2);
}

TEST_CASE("modular constructors accept every integral width", "[nt][accepted_calls]") {
	using Z = atl::mod_int32<7>;
	using D = atl::dynamic_mod_int32<91>;
	D::set_mod(7);
	auto check = []<atl::integral Integer>(Integer value) {
		REQUIRE(Z(value).value() == atl::reduce_mod(value, 7u));
		REQUIRE(D(value).value() == Z(value).value());
		REQUIRE(Z::unchecked(Integer(3)).value() == 3);
		REQUIRE(D::unchecked(Integer(3)).value() == 3);
	};
	check(atl::int8(-1));
	check(atl::uint8(6));
	check(atl::int16(-1));
	check(atl::uint16(6));
	check(atl::int32(-1));
	check(atl::uint32(6));
	check(atl::int64(-1));
	check(atl::uint64(6));
	check(atl::int128(-1));
	check(atl::uint128(6));
	check(static_cast<unsigned long long>(6));
}

TEST_CASE("modular stream insertion accepts standard word types and uint128", "[nt][accepted_calls]") {
	using D = atl::dynamic_mod_int32<94>;
	D::set_mod(7);
	std::ostringstream output;
	output << atl::mod_int32<7>(3) << ' ' << atl::mod_int64<18446744073709551557ull>(-1) << ' ' << D(3) << ' ' << atl::mod_int<atl::uint128, (atl::uint128(1) << 89) - 1>(atl::uint128(1) << 64);
	REQUIRE(output.str() == "3 18446744073709551556 3 18446744073709551616");
}

namespace x2_accepted {
	struct nonconst_rule {
		atl::uint64 operator()(atl::uint64, atl::uint64 e, atl::uint64, std::vector<atl::uint64> const &) {
			return e + 1;
		}
	};

	struct const_rule {
		constexpr atl::uint64 operator()(
			atl::uint64, atl::uint64 e, atl::uint64, std::vector<atl::uint64> const &) const {
			return e + 1;
		}
	};

	struct arithmetic_value;

	struct arithmetic_step {
		unsigned value;
		constexpr operator arithmetic_value() const;
	};

	struct arithmetic_value {
		unsigned value = 0;
		constexpr arithmetic_value() = default;

		constexpr arithmetic_value(atl::uint64 v) : value(static_cast<unsigned>(v)) {}

		friend constexpr arithmetic_value operator*(arithmetic_value a, arithmetic_value b) {
			return a.value * b.value;
		}

		friend constexpr arithmetic_value operator+(arithmetic_value a, int b) {
			return a.value + static_cast<unsigned>(b);
		}

		friend constexpr arithmetic_step operator-(arithmetic_value a, int b) {
			return {a.value - static_cast<unsigned>(b)};
		}

		friend constexpr arithmetic_value operator*(arithmetic_value const &a, arithmetic_step &b) {
			return a.value * b.value;
		}

		friend arithmetic_value operator*(arithmetic_value, arithmetic_step &&) = delete;
	};

	constexpr arithmetic_step::operator arithmetic_value() const { return value; }

	static_assert(arithmetic_value(arithmetic_step{3}).value == 3);

	struct accumulating_value {
		unsigned value;

		constexpr accumulating_value operator()(accumulating_value rhs) const { return {value + rhs.value}; }
	};

	struct explicit_bool {
		bool value;

		explicit constexpr operator bool() const { return value; }
	};

	struct sum_result {
		unsigned value;

		constexpr operator unsigned() const { return value; }
	};
}

TEST_CASE("number theory accepts stateful and wrapped callables", "[nt][accepted_calls]") {
	x2_accepted::nonconst_rule rule;
	x2_accepted::const_rule const readonly;
	auto const expected = atl::divisor_count_range(13);
	auto const verify = [&]<atl::partial_multiplicative_function<atl::uint64> Rule>(Rule &&fn) {
		auto const values = atl::multiplicative_fn_range<atl::uint64>(13, fn);
		for (std::size_t i = 0; i < values.size(); ++i) REQUIRE(values[i] == expected[i]);
		REQUIRE(atl::experimental::multiplicative_fn_range_by_fused_sieve<atl::uint64>(13, fn) == values);
	};
	verify(rule);
	verify(std::ref(rule));
	verify(readonly);
	verify(std::cref(readonly));
	verify([bias = atl::uint64(1)](atl::uint64, atl::uint64 e, atl::uint64, std::vector<atl::uint64> const &) mutable {
		return e + bias;
	});
	REQUIRE(atl::divisor_power_sum_range<atl::mod_int32<7>>(13, 2)[12].value() == 0);
	REQUIRE(atl::divisor_power_sum_range<x2_accepted::arithmetic_value>(13, 2)[12].value == 210);

	auto const inspect_engine = [bias = atl::uint64(0)]<atl::detail::modular_multiplier Engine>(Engine const &engine) {
		return engine.mod() + bias;
	};
	REQUIRE(atl::detail::with_engine(17, std::cref(inspect_engine)) == 17);
	auto const add = [bias = 0u](unsigned l, unsigned r) { return l + r + bias; };
	REQUIRE(atl::floor_staircase_product(5, 3, 2, 1, 1u, 10u, 0u, std::cref(add)) == 53);
	unsigned call_count = 0;
	auto mutable_add = [&call_count](unsigned l, unsigned r) mutable { ++call_count; return l + r; };
	REQUIRE(atl::floor_staircase_product(5, 3, 2, 1, 1u, 10u, 0u, std::ref(mutable_add)) == 53);
	REQUIRE(call_count > 0);
	using value = x2_accepted::accumulating_value;
	REQUIRE(atl::floor_staircase_product(5, 3, 2, 1, value{1}, value{10}, value{0}, &value::operator()).value == 53);
}

TEST_CASE("staircase operations may return references or convertible results", "[nt][accepted_calls]") {
	using value = std::vector<unsigned>;
	auto const minimum = [](value const &l, value const &r) -> value const & { return (std::min)(l, r); };
	value const identity{1000};
	value const up{1, 2, 3}, right{4, 5, 6};
	REQUIRE(atl::detail::monoid_power(up, 19, identity, minimum) == up);
	REQUIRE(atl::floor_staircase_product(5, 3, 2, 1, up, right, identity, minimum) == up);
	REQUIRE(atl::floor_staircase_product(0, 3, 2, 1, up, right, identity, minimum) == identity);
	auto const add = [](unsigned l, unsigned r) { return x2_accepted::sum_result{l + r}; };
	REQUIRE(atl::floor_staircase_product(5, 3, 2, 1, 1u, 10u, 0u, add) == 53);
}

TEST_CASE("factorization predicates may return an explicit bool", "[nt][accepted_calls]") {
	auto predicate = [](atl::uint64 n) mutable { return x2_accepted::explicit_bool{atl::is_prime(n)}; };
	auto const split = [](atl::uint64 n) { return atl::detail::nontrivial_factor(n); };
	for (atl::uint64 n : {atl::uint64(1), atl::uint64(2), atl::uint64(45), atl::uint64(1009 * 1013)}) {
		std::vector<atl::prime_factor> factors;
		atl::experimental::factor_cofactor(n, factors, std::ref(predicate), split);
		REQUIRE(factors == atl::factorization(n));
	}
}

TEST_CASE("fused sieve callbacks read only guaranteed completed entries", "[nt][accepted_calls]") {
	auto const rule = [](atl::uint64 p, atl::uint64 e, atl::uint64 v, std::vector<atl::uint64> const &pre) {
		return e == 1 ? pre[static_cast<std::size_t>(p - 1)] + 1 : pre[static_cast<std::size_t>(v / p)] * p;
	};
	for (std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(5), std::size_t(13),
						  std::size_t(1000)}) {
		auto const values = atl::multiplicative_fn_range<atl::uint64>(n, rule);
		REQUIRE(atl::experimental::multiplicative_fn_range_by_fused_sieve<atl::uint64>(n, rule) == values);
		for (std::size_t i = 0; i < n; ++i) REQUIRE(values[i] == i);
	}
	// The library's ascending pass also permits the entire smaller prefix.
	auto const prefix_rule = [](atl::uint64, atl::uint64, atl::uint64 v, std::vector<atl::uint64> const &pre) {
		return pre[static_cast<std::size_t>(v - 1)] + 1;
	};
	REQUIRE(atl::multiplicative_fn_range<atl::uint64>(5, prefix_rule) == std::vector<atl::uint64>{0, 1, 2, 3, 4});
}

TEST_CASE("congruences accept input views and software-width integers", "[nt][accepted_calls]") {
	std::vector<atl::congruence<atl::uint128>> const classes{{2, 3}, {3, 5}};
	auto view = classes | std::views::filter([](atl::congruence<atl::uint128> const &) { return true; });
	REQUIRE(atl::chinese_remainder(view) == atl::congruence<atl::uint128>{8, 15});
	REQUIRE(atl::chinese_remainder_mod(view, atl::uint128(7)) == 1);
	REQUIRE(atl::solve_linear_congruence(atl::int128(-3), atl::uint128(2), atl::uint128(7))
			== atl::congruence<atl::uint128>{4, 7});
}

TEST_CASE("nt transform accepted calls", "[nt][constraints]") {
	using TransformElement = atl::mod_int<atl::uint32, 998244353>;
	std::array<TransformElement, 4> values{1, 2, 3, 4};
	std::span<TransformElement> mutable_view(values);
	std::span<TransformElement const> const_view(values);
	auto const original = values;
	atl::ntt(mutable_view);
	atl::inverse_ntt(mutable_view);
	CHECK(values == original);
	CHECK(atl::convolution(const_view, original).size() == 7);
	CHECK(atl::convolution_uint64(std::array<int, 2>{1, 2}, std::array<atl::uint128, 1>{3}) == std::vector<atl::uint64>{3, 6});
	std::array<atl::cplx, 2> complex_values{{{1, 0}, {2, 0}}};
	atl::fft(std::span(complex_values));
	atl::inverse_fft(std::span(complex_values));
	CHECK(std::abs(complex_values[0].re - 1) < 1e-12);
	atl::subset_zeta(mutable_view);
	atl::subset_mobius(mutable_view);
	atl::superset_zeta(mutable_view);
	atl::superset_mobius(mutable_view);
	atl::walsh_hadamard(mutable_view);
	atl::inverse_walsh_hadamard(mutable_view);
	CHECK(values == original);
	CHECK(atl::or_convolution(const_view, original).size() == 4);
	CHECK(atl::and_convolution(const_view, original).size() == 4);
	CHECK(atl::xor_convolution(const_view, original).size() == 4);
	CHECK(atl::subset_convolution(const_view, original).size() == 4);
	std::vector<TransformElement> raw_source(64, 1);
	auto get = [&raw_source](std::size_t i) { return raw_source[i].value(); };
	auto const wrapped_get = std::cref(get);
	auto const raw = atl::detail::raw_convolution<atl::scalar_ntt, TransformElement>(64, wrapped_get, 64, get);
	CHECK(raw[63] == 64);
}

TEST_CASE("nt transform contenders accept views and callable wrappers", "[nt][constraints]") {
	using Element = atl::mod_int<atl::uint32, 998244353>;
	std::array<Element, 4> values{1, 2, 3, 4};
	auto const original = values;
	std::span<Element> view(values);
	atl::experimental::ntt_by_radix2(view, false);
	atl::experimental::ntt_by_radix2(view, true);
	CHECK(values == original);
	atl::experimental::ntt_by_dif(view, false);
	atl::experimental::ntt_by_dif(view, true);
	CHECK(values == original);
	CHECK(atl::experimental::convolution_by_radix2(view, original) == atl::convolution(view, original));
	CHECK(atl::experimental::convolution_by_dif(view, original) == atl::convolution(view, original));
	CHECK(atl::experimental::convolution_by_lazy_montgomery(view, original) == atl::convolution(view, original));
	CHECK(atl::experimental::convolution_by_radix4(view, original) == atl::convolution(view, original));
	unsigned calls = 0;
	auto butterfly = [&calls](Element &lo, Element &hi) mutable {
		++calls;
		hi += lo;
	};
	atl::experimental::transform_by_bit_outer_radix2(view.begin(), 2, std::ref(butterfly));
	atl::subset_mobius(view);
	CHECK(values == original);
	atl::experimental::transform_by_radix4(view.begin(), 2, std::ref(butterfly));
	atl::subset_mobius(view);
	CHECK(values == original);
	atl::experimental::transform_by_blocked_radix4(view.begin(), 2, std::ref(butterfly));
	atl::subset_mobius(view);
	CHECK(values == original);
	CHECK(calls == 12);
	CHECK(atl::experimental::subset_convolution_by_rank_major(view, original) == atl::subset_convolution(view, original));
}

namespace {
	struct declining_transform_engine : atl::ntt_engine_base<declining_transform_engine> {
		template<atl::static_mod_int Element>
		static constexpr bool supports() { return false; }

		template<atl::static_mod_int Element>
		static void forward(decltype(Element::mod()) *, atl::uint32) {}

		template<atl::static_mod_int Element>
		static void inverse(decltype(Element::mod()) *, atl::uint32) {}
	};

	static_assert(atl::ntt_engine<declining_transform_engine>);
}

TEST_CASE("nt engine capability fallback remains accepted", "[nt][constraints]") {
	using Element = atl::mod_int<atl::uint32, 998244353>;
	std::vector<Element> const values(64, 1);
	CHECK(atl::convolution<declining_transform_engine>(values, values) == atl::convolution(values, values));
}

#include <atl/experimental/nt/linear_recurrence_variants.hpp>
#include <atl/experimental/nt/matrix_variants.hpp>
#include <atl/nt/linear_recurrence.hpp>
#include <atl/nt/matrix.hpp>
#include <iterator>

namespace {
	struct policy_value {
		int value = 0;
		friend constexpr bool operator==(policy_value const &, policy_value const &) = default;
	};

	struct policy_without_operators {
		using value_type = policy_value;

		static constexpr value_type zero() { return {}; }

		static constexpr value_type one() { return {1}; }

		static constexpr value_type add(value_type a, value_type b) { return {a.value + b.value}; }

		static constexpr value_type multiply(value_type a, value_type b) { return {a.value * b.value}; }
	};
}

TEST_CASE("matrix accepts policies without element arithmetic", "[nt][constraints]") {
	using Matrix = atl::matrix<policy_value, policy_without_operators>;
	Matrix a{{{1}, {2}}, {{3}, {4}}};
	a += Matrix(2, 2);
	a *= Matrix::identity(2);
	CHECK((a + a)(0, 1).value == 4);
	CHECK((a * a)(0, 0).value == 7);
	std::array<policy_value, 2> const v{{{1}, {1}}};
	CHECK((a * std::span<policy_value const>(v))[1].value == 7);
	CHECK(atl::power(a, 2u)(0, 0).value == 7);
	CHECK(a.transpose()(0, 1).value == 3);
	CHECK(Matrix::identity(2) * a == a);
}

TEST_CASE("recurrences accept const views and distinct sentinels", "[nt][constraints]") {
	using Z = atl::mod_int32<998244353>;
	std::array<Z, 8> const fib{0, 1, 1, 2, 3, 5, 8, 13};
	std::span<Z const> const view(fib);
	auto const noncommon = std::ranges::subrange(std::counted_iterator(fib.begin(), 8), std::default_sentinel);
	CHECK(atl::berlekamp_massey(view) == std::vector<Z>{1, 1});
	CHECK(atl::berlekamp_massey(noncommon) == std::vector<Z>{1, 1});
	std::array<Z, 2> const coefficients{1, 1};
	CHECK(atl::linear_recurrence_term(coefficients, noncommon, 10) == Z(55));
	std::array<Z, 1> const numerator{1};
	std::array<Z, 2> const denominator{1, -2};
	CHECK(atl::rational_series_coefficient(numerator, denominator, 10) == Z(1024));
	CHECK(atl::rational_series_coefficient(noncommon, denominator, 0) == Z(0));
}

TEST_CASE("matrix lazy contenders accept their boundary moduli", "[nt][constraints]") {
	using Z = atl::mod_int32<2>;
	std::array<Z, 4> const values{1, 1, 1, 1};
	std::span<Z const> const input(values);
	std::vector<Z> const expected(4, Z(0));
	CHECK(atl::detail::lazy_product<Z>(input, input, 2, 2, 2) == expected);
	CHECK(atl::experimental::matrix_product_by_lazy_transpose<Z>(input, input, 2) == expected);
	CHECK(atl::experimental::matrix_product_by_lazy_tiles<Z, 1>(input, input, 2) == expected);
	CHECK(atl::experimental::matrix_product_by_lazy_subtract<Z>(input, input, 2) == expected);
}

namespace {
	template<atl::semiring Policy>
		requires(atl::integral<typename Policy::value_type> || std::floating_point<typename Policy::value_type>)
	constexpr bool numeric_policy_holds() {
		using Value = typename Policy::value_type;
		Value const x = [] {
			if constexpr (atl::integral<Value> && sizeof(Value) > sizeof(atl::uint64))
				return Value((Value(1) << 100) + 7);
			else
				return Value(7);
		}();
		Value const y = Value(x + Value(4));
		if (Policy::add(Policy::zero(), x) != x || Policy::add(x, Policy::zero()) != x
			|| Policy::multiply(Policy::one(), x) != x || Policy::multiply(x, Policy::one()) != x
			|| Policy::multiply(Policy::zero(), x) != Policy::zero()
			|| Policy::multiply(x, Policy::zero()) != Policy::zero()) return false;
		atl::matrix<Value, Policy> const a{{x, y}, {y, x}};
		return a * atl::matrix<Value, Policy>::identity(2) == a;
	}

	static_assert(numeric_policy_holds<atl::min_plus_semiring<atl::int128>>());
	static_assert(numeric_policy_holds<atl::min_plus_semiring<atl::uint128>>());
	static_assert(numeric_policy_holds<atl::max_plus_semiring<atl::int128>>());
	static_assert(numeric_policy_holds<atl::max_min_semiring<atl::int128>>());
	static_assert(numeric_policy_holds<atl::max_min_semiring<atl::uint128>>());
}

TEST_CASE("numeric semirings accept portable wide integers", "[nt][constraints]") {
	CHECK(numeric_policy_holds<atl::min_plus_semiring<atl::int128>>());
	CHECK(numeric_policy_holds<atl::min_plus_semiring<atl::uint128>>());
	CHECK(numeric_policy_holds<atl::max_plus_semiring<atl::int128>>());
	CHECK(numeric_policy_holds<atl::max_min_semiring<atl::int128>>());
	CHECK(numeric_policy_holds<atl::max_min_semiring<atl::uint128>>());
	CHECK(atl::min_plus_semiring<atl::int128>::multiply(7, 11) == atl::int128(18));
	CHECK(atl::min_plus_semiring<atl::uint128>::multiply(7, 11) == atl::uint128(18));
	CHECK(atl::max_plus_semiring<atl::int128>::multiply(-7, 11) == atl::int128(4));
	CHECK(atl::max_min_semiring<atl::int128>::multiply(-7, 11) == atl::int128(-7));
	CHECK(atl::max_min_semiring<atl::uint128>::multiply(7, 11) == atl::uint128(7));
	CHECK(numeric_policy_holds<atl::min_plus_semiring<double>>());
	CHECK(numeric_policy_holds<atl::max_plus_semiring<double>>());
	CHECK(numeric_policy_holds<atl::max_min_semiring<double>>());
	CHECK(numeric_policy_holds<atl::plus_times_semiring<atl::uint8>>());
}

namespace {
	using series_mint = atl::mod_int32<998244353>;
	using series_vector = std::vector<series_mint>;

	template<atl::prime_mod_int_range Unit, atl::prime_mod_int_range Zero>
	series_vector series_inputs_agree(Unit const &unit, Zero const &zero) {
		auto const inverse = atl::series_inverse(unit, 5);
		CHECK(atl::series_logarithm(unit, 5) == atl::series_logarithm(series_vector{1, 1}, 5));
		CHECK(atl::series_exponential(zero, 5) == atl::series_exponential(series_vector{0, 1}, 5));
		CHECK(atl::series_division(unit, unit, 5) == series_vector{1, 0, 0, 0, 0});
		CHECK(atl::series_power(unit, 3, 5) == series_vector{1, 3, 3, 1, 0});
		CHECK(atl::series_inverse_sparse(unit, 5) == inverse);
		CHECK(atl::series_division_sparse(unit, unit, 5) == series_vector{1, 0, 0, 0, 0});
		CHECK(atl::series_logarithm_sparse(unit, 5) == atl::series_logarithm(unit, 5));
		CHECK(atl::series_exponential_sparse(zero, 5) == atl::series_exponential(zero, 5));
		CHECK(atl::series_power_sparse(unit, 3, 5) == series_vector{1, 3, 3, 1, 0});
		CHECK(atl::series_square_root(unit, 5) == atl::series_square_root_sparse(unit, 5));
		CHECK(atl::experimental::series_inverse_by_recurrence(unit, 5) == inverse);
		CHECK(atl::experimental::series_inverse_by_plain_newton(unit, 5) == inverse);
		CHECK(atl::experimental::series_inverse_by_newton_driver(unit, 5) == inverse);
		CHECK(atl::experimental::series_division_by_inverse_product(unit, unit, 5) == series_vector{1, 0, 0, 0, 0});
		CHECK(atl::experimental::series_logarithm_by_inverse_product(unit, 5) == atl::series_logarithm(unit, 5));
		CHECK(atl::experimental::series_exponential_by_recurrence(zero, 5) == atl::series_exponential(zero, 5));
		CHECK(atl::experimental::series_exponential_by_plain_newton(zero, 5) == atl::series_exponential(zero, 5));
		CHECK(atl::experimental::series_exponential_by_newton_driver(zero, 5) == atl::series_exponential(zero, 5));
		CHECK(atl::experimental::series_power_by_binary_powering(unit, 3, 5) == series_vector{1, 3, 3, 1, 0});
		CHECK(atl::experimental::series_square_root_by_plain_newton(unit, 5) == atl::series_square_root(unit, 5));
		CHECK(atl::experimental::series_square_root_by_logarithm(unit, 5) == atl::series_square_root(unit, 5));
		CHECK(atl::polynomial_division(unit, unit).quotient == series_vector{1});
		CHECK(atl::experimental::polynomial_division_by_schoolbook(unit, unit).quotient == series_vector{1});
		return inverse;
	}
}

TEST_CASE("power series entries accept every const input range and engine", "[nt][constraints]") {
	series_vector const unit{1, 1}, zero{0, 1};
	auto const expected = series_inputs_agree(unit, zero);
	CHECK(expected == series_vector{1, -1, 1, -1, 1});
	std::array<series_mint, 2> const unit_array{1, 1}, zero_array{0, 1};
	CHECK(series_inputs_agree(unit_array, zero_array) == expected);
	std::forward_list<series_mint> const unit_list{1, 1}, zero_list{0, 1};
	CHECK(series_inputs_agree(unit_list, zero_list) == expected);
	std::list<series_mint> const unit_bidirectional{1, 1}, zero_bidirectional{0, 1};
	CHECK(series_inputs_agree(unit_bidirectional, zero_bidirectional) == expected);
	atl_test::input_only<series_mint> const unit_single_pass(unit), zero_single_pass(zero);
	CHECK(series_inputs_agree(unit_single_pass, zero_single_pass) == expected);
	auto const unit_view = std::views::iota(0, 2) | std::views::transform([](int) { return series_mint(1); });
	auto const zero_view = std::views::iota(0, 2) | std::views::transform([](int i) { return series_mint(i); });
	CHECK(series_inputs_agree(unit_view, zero_view) == expected);
	CHECK(series_inputs_agree(std::span<series_mint const>(unit), std::span<series_mint const>(zero)) == expected);
	CHECK(atl::series_inverse<atl::scalar_ntt>(unit, 5) == expected);
	CHECK(atl::series_division<atl::scalar_ntt>(unit_array, unit_list, 5) == series_vector{1, 0, 0, 0, 0});
	CHECK(atl::series_division_sparse(unit_array, unit_list, 5) == series_vector{1, 0, 0, 0, 0});
	CHECK(atl::polynomial_division<atl::scalar_ntt>(unit_array, unit_list).quotient == series_vector{1});
	using wide_mint = atl::mod_int64<18446744073709551557ull>;
	CHECK(atl::series_inverse(std::vector<wide_mint>{1, 1}, 3) == std::vector<wide_mint>{1, -1, 1});
}

TEST_CASE("prime-field helpers accept every const input range and modulus", "[nt][constraints]") {
	series_vector const values{2, 4, 8};
	std::array<series_mint, 3> const array{2, 4, 8};
	std::forward_list<series_mint> const list{2, 4, 8};
	atl_test::input_only<series_mint> const single_pass(values);
	CHECK(atl::batch_inverse(array) == atl::batch_inverse(values));
	CHECK(atl::batch_inverse(list) == atl::batch_inverse(values));
	CHECK(atl::batch_inverse(single_pass) == atl::batch_inverse(values));
	CHECK(atl::series_derivative(list) == atl::series_derivative(values));
	CHECK(atl::series_derivative(single_pass) == atl::series_derivative(values));
	CHECK(atl::series_integral(list) == atl::series_integral(values));
	CHECK(atl::series_integral(single_pass) == atl::series_integral(values));
	CHECK(atl::factorial_table<atl::mod_int32<2>>(1).factorial(1) == atl::mod_int32<2>(1));
	CHECK(atl::factorial_table<atl::mod_int64<18446744073709551557ull>>(3).factorial(3) == atl::mod_int64<18446744073709551557ull>(6));
	series_mint const square(49);
	CHECK(atl::square_root_mod(square) == series_mint(7));
	CHECK(atl::experimental::square_root_mod_by_cipolla(square) == series_mint(7));
	CHECK(atl::square_root_mod(atl::mod_int64<18446744073709551557ull>(4)) == atl::mod_int64<18446744073709551557ull>(2));
	CHECK(atl::square_root_mod(std::uint64_t{4}, std::uint64_t{17}) == 2);
}

namespace {
	using points_field = atl::mod_int32<998244353>;
	using points_vector = std::vector<points_field>;

	/** @brief Every point operation and contender gives the result of the vector call, whatever the input ranges are. */
	template<atl::prime_mod_int_range Polynomial, atl::prime_mod_int_range Points, atl::prime_mod_int_range Values>
	void point_inputs_agree(Polynomial const &polynomial, Points const &points, Values const &values) {
		namespace ex = atl::experimental;
		points_vector const reference_polynomial{1, 2, 3}, reference_points{4, 5, 6}, reference_values{57, 86, 121};
		points_field const shift(7), first(2), ratio(3);
		auto const shifted = atl::polynomial_taylor_shift(reference_polynomial, shift);
		auto const evaluated = atl::polynomial_evaluation(reference_polynomial, reference_points);
		CHECK(evaluated == reference_values);
		CHECK(atl::polynomial_taylor_shift(polynomial, shift) == shifted);
		CHECK(ex::polynomial_taylor_shift_by_divide_and_conquer(polynomial, shift) == shifted);
		CHECK(ex::polynomial_taylor_shift_by_horner(polynomial, shift) == shifted);
		auto const from_roots = atl::polynomial_from_roots(reference_points);
		CHECK(atl::polynomial_from_roots(points) == from_roots);
		CHECK(ex::polynomial_from_roots_by_sequential_product(points) == from_roots);
		CHECK(atl::polynomial_evaluation(polynomial, points) == evaluated);
		CHECK(ex::polynomial_evaluation_by_remainder_tree(polynomial, points) == evaluated);
		CHECK(ex::polynomial_evaluation_by_horner(polynomial, points) == evaluated);
		CHECK(ex::polynomial_evaluation_by_leaf_cutoff(polynomial, points, 16) == evaluated);
		CHECK(atl::polynomial_interpolation(points, values) == reference_polynomial);
		CHECK(ex::polynomial_interpolation_by_remainder_tree(points, values) == reference_polynomial);
		CHECK(ex::polynomial_interpolation_by_newton_differences(points, values) == reference_polynomial);
		auto const geometric = atl::polynomial_geometric_evaluation(reference_polynomial, first, ratio, 5);
		CHECK(atl::polynomial_geometric_evaluation(polynomial, first, ratio, 5) == geometric);
		CHECK(ex::polynomial_geometric_evaluation_by_horner(polynomial, first, ratio, 5) == geometric);
		auto const sampled = atl::sample_point_shift(reference_values, shift, 5);
		CHECK(atl::sample_point_shift(values, shift, 5) == sampled);
		CHECK(ex::sample_point_shift_by_falling_factorials(values, shift, 5) == sampled);
		CHECK(ex::sample_point_shift_by_direct_barycentric(values, shift, 5) == sampled);
		atl::polynomial_point_tree<points_field> const tree(points);
		CHECK(tree.point_count() == 3);
		CHECK(tree.evaluate(polynomial) == evaluated);
		CHECK(tree.interpolate(values) == reference_polynomial);
	}

	/** @brief Composition, power projection and reversion, with every contender, agree with the vector call. */
	template<atl::prime_mod_int_range Outer, atl::prime_mod_int_range Inner>
	void composition_inputs_agree(Outer const &outer, Inner const &inner) {
		namespace ex = atl::experimental;
		points_vector const reference_outer{1, 2, 3}, reference_inner{0, 1, 1};
		auto const composed = atl::series_composition(reference_outer, reference_inner, 5);
		CHECK(composed == points_vector{1, 2, 5, 6, 3});
		CHECK(atl::series_composition(outer, inner, 5) == composed);
		CHECK(ex::series_composition_by_brent_kung(outer, inner, 5) == composed);
		CHECK(ex::series_composition_by_fast_horner(outer, inner, 5) == composed);
		CHECK(ex::series_composition_by_schoolbook_horner(outer, inner, 5) == composed);
		CHECK(ex::series_composition_by_cutoffs(outer, inner, 5, 1, 1) == composed);
		auto const projected = atl::series_power_projection(reference_outer, reference_inner, 5);
		CHECK(atl::series_power_projection(outer, inner, 5) == projected);
		CHECK(ex::series_power_projection_by_padded_bostan_mori(outer, inner, 5) == projected);
		CHECK(ex::series_power_projection_by_repeated_products(outer, inner, 5) == projected);
		CHECK(ex::series_power_projection_by_bostan_mori(outer, inner, 5) == projected);
		auto const inverse = atl::series_compositional_inverse(reference_inner, 5);
		CHECK(atl::series_composition(reference_inner, inverse, 5) == points_vector{0, 1, 0, 0, 0});
		CHECK(atl::series_compositional_inverse(inner, 5) == inverse);
		CHECK(ex::series_compositional_inverse_by_composition_newton(inner, 5) == inverse);
		CHECK(ex::series_compositional_inverse_by_projection_lagrange(inner, 5) == inverse);
		CHECK(ex::series_compositional_inverse_by_baby_giant_lagrange(inner, 5) == inverse);
		CHECK(ex::series_compositional_inverse_by_coefficient_recurrence(inner, 5) == inverse);
	}
}

TEST_CASE("polynomial point operations accept every const input range", "[nt][constraints]") {
	points_vector const polynomial{1, 2, 3}, points{4, 5, 6}, values{57, 86, 121};
	point_inputs_agree(polynomial, points, values);
	std::array<points_field, 3> const polynomial_array{1, 2, 3}, points_array{4, 5, 6}, values_array{57, 86, 121};
	point_inputs_agree(polynomial_array, points_array, values_array);
	std::forward_list<points_field> const polynomial_list{1, 2, 3}, points_list{4, 5, 6}, values_list{57, 86, 121};
	point_inputs_agree(polynomial_list, points_list, values_list);
	std::list<points_field> const polynomial_bidirectional{1, 2, 3}, points_bidirectional{4, 5, 6}, values_bidirectional{57, 86, 121};
	point_inputs_agree(polynomial_bidirectional, points_bidirectional, values_bidirectional);
	atl_test::input_only<points_field> const polynomial_single_pass(polynomial), points_single_pass(points), values_single_pass(values);
	point_inputs_agree(polynomial_single_pass, points_single_pass, values_single_pass);
	auto const polynomial_view = std::views::iota(1, 4) | std::views::transform([](int i) { return points_field(i); });
	auto const points_view = std::views::iota(4, 7) | std::views::transform([](int i) { return points_field(i); });
	point_inputs_agree(polynomial_view, points_view, values);
	point_inputs_agree(std::span<points_field const>(polynomial), std::span<points_field const>(points), std::span<points_field const>(values));
}

TEST_CASE("polynomial products accept nested input ranges and every engine", "[nt][constraints]") {
	points_vector const expected{1, 3, 2};
	std::vector<points_vector> const vectors{{1, 1}, {1, 2}};
	std::array<std::array<points_field, 2>, 2> const arrays{{{1, 1}, {1, 2}}};
	std::list<std::forward_list<points_field>> const lists{{1, 1, 0}, {1, 2}};
	std::vector<atl_test::input_only<points_field>> const single_pass{atl_test::input_only<points_field>({1, 1}), atl_test::input_only<points_field>({1, 2})};
	CHECK(atl::polynomial_product(vectors) == expected);
	CHECK(atl::polynomial_product(arrays) == expected);
	CHECK(atl::polynomial_product(lists) == expected);
	CHECK(atl::polynomial_product(single_pass) == expected);
	CHECK(atl::polynomial_product<atl::scalar_ntt>(vectors) == expected);
	CHECK(atl::experimental::polynomial_product_by_smallest_first(lists) == expected);
	CHECK(atl::experimental::polynomial_product_by_sequential_schoolbook(single_pass) == expected);
	points_vector const polynomial{1, 2, 3}, points{4, 5, 6};
	CHECK(atl::polynomial_evaluation<atl::scalar_ntt>(polynomial, points) == points_vector{57, 86, 121});
	CHECK(atl::polynomial_interpolation<atl::scalar_ntt>(points, atl::polynomial_evaluation(polynomial, points)) == polynomial);
	CHECK(atl::polynomial_taylor_shift<atl::scalar_ntt>(polynomial, points_field(1)) == points_vector{6, 8, 3});
	CHECK(atl::polynomial_geometric_evaluation<atl::scalar_ntt>(polynomial, points_field(1), points_field(2), 3) == points_vector{6, 17, 57});
	CHECK(atl::sample_point_shift<atl::scalar_ntt>(points_vector{1, 6, 17}, points_field(3), 2) == points_vector{34, 57});
	using wide_field = atl::mod_int64<18446744073709551557ull>;
	std::vector<wide_field> const wide_polynomial{1, 2, 3}, wide_points{4, 5, 6};
	CHECK(atl::polynomial_evaluation(wide_polynomial, wide_points) == std::vector<wide_field>{57, 86, 121});
	CHECK(atl::polynomial_point_tree<wide_field>(wide_points).interpolate(atl::polynomial_evaluation(wide_polynomial, wide_points)) == wide_polynomial);
}

TEST_CASE("series composition accepts every const input range and engine", "[nt][constraints]") {
	points_vector const outer{1, 2, 3}, inner{0, 1, 1};
	composition_inputs_agree(outer, inner);
	std::array<points_field, 3> const outer_array{1, 2, 3}, inner_array{0, 1, 1};
	composition_inputs_agree(outer_array, inner_array);
	std::forward_list<points_field> const outer_list{1, 2, 3}, inner_list{0, 1, 1};
	composition_inputs_agree(outer_list, inner_list);
	std::list<points_field> const outer_bidirectional{1, 2, 3}, inner_bidirectional{0, 1, 1};
	composition_inputs_agree(outer_bidirectional, inner_bidirectional);
	atl_test::input_only<points_field> const outer_single_pass(outer), inner_single_pass(inner);
	composition_inputs_agree(outer_single_pass, inner_single_pass);
	auto const outer_view = std::views::iota(1, 4) | std::views::transform([](int i) { return points_field(i); });
	auto const inner_view = std::views::iota(0, 3) | std::views::transform([](int i) { return points_field(i == 0 ? 0 : 1); });
	composition_inputs_agree(outer_view, inner_view);
	composition_inputs_agree(std::span<points_field const>(outer), std::span<points_field const>(inner));
	CHECK(atl::series_composition<atl::scalar_ntt>(outer, inner, 5) == points_vector{1, 2, 5, 6, 3});
	CHECK(atl::series_power_projection<atl::scalar_ntt>(outer, inner, 5) == atl::series_power_projection(outer, inner, 5));
	CHECK(atl::series_compositional_inverse<atl::scalar_ntt>(inner, 5) == atl::series_compositional_inverse(inner, 5));
	using wide_field = atl::mod_int64<18446744073709551557ull>;
	std::vector<wide_field> const wide_outer{1, 2, 3}, wide_inner{0, 1, 1};
	CHECK(atl::series_composition(wide_outer, wide_inner, 5) == std::vector<wide_field>{1, 2, 5, 6, 3});
	CHECK(atl::series_composition(wide_inner, atl::series_compositional_inverse(wide_inner, 5), 5) == std::vector<wide_field>{0, 1, 0, 0, 0});
}

namespace {
	using gcd_field = atl::mod_int32<998244353>;
	using gcd_vector = std::vector<gcd_field>;

	/** @brief Every gcd-family function and contender gives the vector result whatever the input ranges are. */
	template<atl::prime_mod_int_range First, atl::prime_mod_int_range Second, atl::prime_mod_int_range Series>
	void gcd_inputs_agree(First const &first, Second const &second, Series const &series) {
		namespace ex = atl::experimental;
		gcd_vector const reference_first{2, 3, 1}, reference_second{3, 4, 1}, reference_series{1, 1, 1, 1};
		auto const expected = atl::polynomial_gcd(reference_first, reference_second);
		auto const extended = atl::polynomial_extended_gcd(reference_first, reference_second);
		CHECK(expected == gcd_vector{1, 1});
		CHECK(atl::polynomial_gcd(first, second) == expected);
		CHECK(ex::polynomial_gcd_by_fused_matrix_products(first, second) == expected);
		CHECK(ex::polynomial_gcd_by_fast_division_euclid(first, second) == expected);
		CHECK(ex::polynomial_gcd_by_schoolbook_euclid(first, second) == expected);
		CHECK(ex::polynomial_gcd_by_leaf_cutoff_plain_products(first, second, 1) == expected);
		CHECK(ex::polynomial_gcd_by_leaf_cutoff_fused_products(first, second, 1) == expected);
		auto const result = atl::polynomial_extended_gcd(first, second);
		CHECK(result.gcd == extended.gcd);
		CHECK(result.first_cofactor == extended.first_cofactor);
		CHECK(result.second_cofactor == extended.second_cofactor);
		CHECK(ex::polynomial_extended_gcd_by_fused_matrix_products(first, second).gcd == expected);
		CHECK(ex::polynomial_extended_gcd_by_fast_division_euclid(first, second).gcd == expected);
		CHECK(ex::polynomial_extended_gcd_by_schoolbook_euclid(first, second).gcd == expected);
		CHECK(ex::polynomial_extended_gcd_by_leaf_cutoff_plain_products(first, second, 1).gcd == expected);
		CHECK(ex::polynomial_extended_gcd_by_leaf_cutoff_fused_products(first, second, 1).gcd == expected);
		auto const approximant = atl::pade_approximant(reference_series, 1, 2);
		REQUIRE(approximant.has_value());
		auto const from_range = atl::pade_approximant(series, 1, 2);
		REQUIRE(from_range.has_value());
		CHECK(from_range->numerator == approximant->numerator);
		CHECK(from_range->denominator == approximant->denominator);
		auto const quadratic = ex::pade_approximant_by_quadratic_euclid(series, 1, 2);
		REQUIRE(quadratic.has_value());
		CHECK(quadratic->denominator == approximant->denominator);
		CHECK(ex::berlekamp_massey_by_half_gcd(series) == ex::berlekamp_massey_by_half_gcd(reference_series));
	}

	/** @brief The inverse of `x` modulo `x^2 + 1` is `-x`, whatever the input ranges are. */
	template<atl::prime_mod_int_range Value, atl::prime_mod_int_range Modulus>
	void inverse_inputs_agree(Value const &value, Modulus const &modulus) {
		auto const inverse = atl::polynomial_inverse_mod(value, modulus);
		REQUIRE(inverse.has_value());
		CHECK(*inverse == gcd_vector{0, gcd_field(-1)});
	}

	/** @brief Every fixed-kernel stream and contender gives the result of the vector kernel, whatever the kernel range is. */
	template<atl::prime_mod_int_range Kernel>
	void kernel_inputs_agree(Kernel const &kernel) {
		namespace ex = atl::experimental;
		gcd_vector const reference_kernel{1, 2, 3};
		atl::online_fixed_convolution<gcd_field> reference(reference_kernel), library(kernel);
		ex::online_fixed_convolution_by_block_cutoff<gcd_field> block_cutoff(kernel, 1);
		ex::online_fixed_convolution_by_uncached_blocks<gcd_field> uncached(kernel, 1);
		ex::online_fixed_convolution_by_direct_sums<gcd_field> direct(kernel);
		ex::online_fixed_convolution_by_karatsuba_blocks<gcd_field> karatsuba(kernel, 1);
		for (int i = 1; i <= 40; ++i) {
			gcd_field const expected = reference.push(gcd_field(i));
			CHECK(library.push(gcd_field(i)) == expected);
			CHECK(block_cutoff.push(gcd_field(i)) == expected);
			CHECK(uncached.push(gcd_field(i)) == expected);
			CHECK(direct.push(gcd_field(i)) == expected);
			CHECK(karatsuba.push(gcd_field(i)) == expected);
		}
	}
}

TEST_CASE("polynomial gcd family accepts every const input range", "[nt][constraints]") {
	gcd_vector const first{2, 3, 1}, second{3, 4, 1}, series{1, 1, 1, 1};
	gcd_inputs_agree(first, second, series);
	std::array<gcd_field, 3> const first_array{2, 3, 1}, second_array{3, 4, 1};
	std::array<gcd_field, 4> const series_array{1, 1, 1, 1};
	gcd_inputs_agree(first_array, second_array, series_array);
	std::forward_list<gcd_field> const first_list{2, 3, 1}, second_list{3, 4, 1}, series_list{1, 1, 1, 1};
	gcd_inputs_agree(first_list, second_list, series_list);
	std::list<gcd_field> const first_bidirectional{2, 3, 1}, second_bidirectional{3, 4, 1}, series_bidirectional{1, 1, 1, 1};
	gcd_inputs_agree(first_bidirectional, second_bidirectional, series_bidirectional);
	atl_test::input_only<gcd_field> const first_single_pass(first), second_single_pass(second), series_single_pass(series);
	gcd_inputs_agree(first_single_pass, second_single_pass, series_single_pass);
	auto const first_view = std::views::iota(2, 5) | std::views::transform([](int i) { return gcd_field(i == 4 ? 1 : i); });
	auto const second_view = std::views::iota(3, 6) | std::views::transform([](int i) { return gcd_field(i == 5 ? 1 : i); });
	gcd_inputs_agree(first_view, second_view, series_list);
	gcd_inputs_agree(std::span<gcd_field const>(first), second_list, std::span<gcd_field const>(series));
	gcd_inputs_agree(first, second_single_pass, series_single_pass);
	std::array<gcd_field, 2> const value{0, 1};
	std::array<gcd_field, 3> const modulus{1, 0, 1};
	inverse_inputs_agree(value, modulus);
	inverse_inputs_agree(std::forward_list<gcd_field>{0, 1}, atl_test::input_only<gcd_field>(gcd_vector{1, 0, 1}));
	inverse_inputs_agree(std::span<gcd_field const>(value), std::list<gcd_field>{1, 0, 1});
}

TEST_CASE("polynomial gcd family accepts every engine and wide fields", "[nt][constraints]") {
	gcd_vector const first{2, 3, 1}, second{3, 4, 1};
	CHECK(atl::polynomial_gcd<atl::scalar_ntt>(first, second) == gcd_vector{1, 1});
	CHECK(atl::polynomial_extended_gcd<atl::scalar_ntt>(first, second).gcd == gcd_vector{1, 1});
	CHECK(atl::polynomial_inverse_mod<atl::scalar_ntt>(gcd_vector{0, 1}, gcd_vector{1, 0, 1}).has_value());
	CHECK(atl::pade_approximant<atl::scalar_ntt>(gcd_vector{1, 1, 1, 1}, 1, 2).has_value());
	using wide_field = atl::mod_int64<18446744073709551557ull>;
	std::vector<wide_field> const wide_first{2, 3, 1}, wide_second{3, 4, 1};
	CHECK(atl::polynomial_gcd(wide_first, wide_second) == std::vector<wide_field>{1, 1});
	CHECK(atl::polynomial_extended_gcd(wide_first, wide_second).gcd == std::vector<wide_field>{1, 1});
	CHECK(atl::pade_approximant(std::vector<wide_field>{1, 1, 1, 1}, 1, 2).has_value());
	atl::online_convolution<gcd_field, atl::scalar_ntt> scalar_stream;
	CHECK(scalar_stream.push(gcd_field(2), gcd_field(3)) == gcd_field(6));
	atl::online_fixed_convolution<wide_field> wide_stream(wide_first);
	CHECK(wide_stream.push(wide_field(1)) == wide_field(2));
	CHECK(wide_stream.push(wide_field(1)) == wide_field(5));
}

TEST_CASE("online fixed convolution accepts every const kernel range", "[nt][constraints]") {
	gcd_vector const kernel{1, 2, 3};
	kernel_inputs_agree(kernel);
	kernel_inputs_agree(std::array<gcd_field, 3>{1, 2, 3});
	kernel_inputs_agree(std::forward_list<gcd_field>{1, 2, 3});
	kernel_inputs_agree(std::list<gcd_field>{1, 2, 3, 0, 0});
	kernel_inputs_agree(atl_test::input_only<gcd_field>(kernel));
	kernel_inputs_agree(std::span<gcd_field const>(kernel));
	kernel_inputs_agree(std::views::iota(1, 4) | std::views::transform([](int i) { return gcd_field(i); }));
}

namespace {
	using decision_field = atl::mod_int32<998244353>;
	using decision_vector = std::vector<decision_field>;

	/** @brief A function constrained only by the public range concept, returning the public coefficient alias. */
	template<atl::prime_mod_int_range Range>
	atl::series_coefficient_t<Range> first_coefficient(Range const &range) {
		return *std::ranges::begin(range);
	}
}

TEST_CASE("public prime-field concepts accept references, const ranges and const elements", "[nt][constraints]") {
	decision_vector values{3, 1, 2};
	decision_vector const &const_values = values;
	std::array<decision_field const, 2> const const_elements{decision_field(5), decision_field(6)};
	CHECK(first_coefficient(values) == decision_field(3));
	CHECK(first_coefficient(const_values) == decision_field(3));
	CHECK(first_coefficient(const_elements) == decision_field(5));
	CHECK(first_coefficient(std::span<decision_field const>(values)) == decision_field(3));
	CHECK(first_coefficient(atl_test::input_only<decision_field>(values)) == decision_field(3));
	CHECK(atl::series_inverse(const_elements, 2) == atl::series_inverse(decision_vector{5, 6}, 2));
	CHECK(atl::series_power_sparse(const_elements, 2, 3) == atl::series_power(decision_vector{5, 6}, 2, 3));
}

TEST_CASE("nt-3 interface additions accept const and wrapped inputs", "[nt][constraints]") {
	atl::factorial_table<decision_field> table(2);
	table.extend(10);
	CHECK(table.factorial(10) == decision_field(3628800));
	std::forward_list<decision_field> const points{4, 4, 1};
	atl::polynomial_point_tree<decision_field> const tree(points);
	CHECK(std::ranges::equal(tree.points(), decision_vector{4, 4, 1}));
	auto const &tree_reference = tree;
	CHECK(tree_reference.points().size() == 3);
	auto const result = atl::polynomial_extended_gcd(decision_vector{2, 3, 1}, decision_vector{3, 4, 1});
	auto const copy = result;
	CHECK(result == copy);
	CHECK(std::equality_comparable<atl::polynomial_extended_gcd_result<decision_field>>);
	atl::online_convolution<decision_field, atl::scalar_ntt> stream;
	stream.reserve(std::size_t{8});
	CHECK(stream.push(decision_field(2), decision_field(3)) == decision_field(6));
	atl::online_fixed_convolution<decision_field> fixed(std::array<decision_field, 2>{1, 2});
	fixed.reserve(3u);
	CHECK(fixed.push(decision_field(1)) == decision_field(1));
}

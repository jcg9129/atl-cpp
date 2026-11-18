#include <atl/experimental/nt/bitwise_convolution_variants.hpp>
#include <atl/experimental/nt/convolution_variants.hpp>
#include <atl/experimental/nt/factorization_variants.hpp>
#include <atl/experimental/nt/sieve_variants.hpp>
#include <atl/nt/bitwise_convolution.hpp>
#include <atl/nt/convolution.hpp>
#include <atl/nt/crt.hpp>
#include <atl/nt/discrete_log.hpp>
#include <atl/nt/dynamic_mod_int.hpp>
#include <atl/nt/fft.hpp>
#include <atl/nt/floor_sum.hpp>
#include <atl/nt/gcd.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/modular.hpp>
#include <atl/nt/montgomery.hpp>
#include <atl/nt/multiplicative.hpp>
#include <atl/nt/power.hpp>
#include <atl/nt/static_modulus.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cmath>
#include <concepts>
#include <functional>
#include <ranges>
#include <span>
#include <sstream>
#include <utility>
#include <vector>

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

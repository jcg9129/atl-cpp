#include <atl/experimental/nt/factorization_variants.hpp>
#include <atl/experimental/nt/sieve_variants.hpp>
#include <atl/nt/crt.hpp>
#include <atl/nt/discrete_log.hpp>
#include <atl/nt/dynamic_mod_int.hpp>
#include <atl/nt/floor_sum.hpp>
#include <atl/nt/gcd.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/modular.hpp>
#include <atl/nt/montgomery.hpp>
#include <atl/nt/multiplicative.hpp>
#include <atl/nt/power.hpp>
#include <atl/nt/static_modulus.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <functional>
#include <ranges>
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

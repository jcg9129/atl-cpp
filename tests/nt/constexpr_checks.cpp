#include <atl/experimental/nt/factorization_trial.hpp>
#include <atl/nt/basic.hpp>
#include <atl/nt/constants.hpp>
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
#include <atl/nt/sieve.hpp>
#include <atl/nt/static_modulus.hpp>
#include <atl/types/alias.hpp>
#include <array>
#include <catch2/catch.hpp>
#include <limits>
#include <vector>

namespace {
	using atl::uint128;
	using Dynamic32 = atl::dynamic_mod_int<>;
	using Dynamic64 = atl::dynamic_mod_int<atl::uint64>;
	static_assert(Dynamic32{}.value() == 0);
	static_assert(+Dynamic32{} == Dynamic32{});
	static_assert(Dynamic64{}.value() == 0);
	static_assert(+Dynamic64{} == Dynamic64{});
	using atl::uint32;
	using atl::uint64;

	static_assert(atl::static_modulus<atl::uint8(251)>::mul(250, 250) == 1);
	static_assert(atl::static_modulus<atl::uint8(127)>::mul(126, 126) == 1);
	static_assert(atl::static_modulus<atl::uint16(65521)>::mul(65520, 65520) == 1);
	static_assert(atl::montgomery<atl::uint8(63)>::mul_lazy(125, 125) % 63
				  == atl::montgomery<atl::uint8(63)>::mul(62, 62));
	static_assert(atl::extended_gcd(uint32(12), uint32(18)).gcd == 6);
	static_assert(atl::extended_gcd(-12, 18).gcd == 6u);
	static_assert(atl::extended_gcd(0u, 0u).gcd == 0);
	static_assert([] {
		uint64 const a = ~uint64(0), b = a - 1;
		auto const [g, x, y] = atl::extended_gcd(a, b);
		return g == 1 && atl::int128(a) * x + atl::int128(b) * y == 1;
	}());
	static_assert([] {
		atl::int128 const lowest = atl::int128(uint128(1) << 127);
		auto const result = atl::extended_gcd(lowest, atl::int128(0));
		return result.gcd == uint128(1) << 127 && result.x == -1 && result.y == 0;
	}());

	static_assert(atl::mod_int<uint32, 7>(3).inverse().value() == 5);
	static_assert((atl::mod_int<uint32, 7>(1) / atl::mod_int<uint32, 7>(3)).value() == 5);
	static_assert(atl::power(atl::mod_int<uint32, 1000000007>(2), 10u).value() == 1024);
	static_assert([] {
		using Z = atl::mod_int<uint32, 1000000007>;
		Z a = 5, b = Z::unchecked(1000000006);
		Z c = a;
		++c;
		c--;
		--c;
		c++;
		return (a + b).value() == 4 && (a - b).value() == 6 && (a * b).value() == 1000000002
			&& (-a).value() == 1000000002
			&& (+a) == a && c == a && a.power(3).value() == 125 && !(a == b)
			&& Z().value() == 0 && Z::mod() == 1000000007;
	}());
	static_assert([] {
		using Z = atl::mod_int64<atl::largest_prime_64>;
		Z const a = -1, b = 3;
		return (a * a).value() == 1 && (a + b).value() == 2 && (b / b).value() == 1 && b.inverse() * b == Z(1)
			&& Z::unchecked(atl::largest_prime_64 - 1) == a;
	}());
	static_assert([] {
		using Z = atl::mod_int<uint128, (uint128(1) << 89) - 1>;
		Z const a = Z(uint128(1) << 88);
		return (a * Z(4)).value() == 2;
	}());
	static_assert([] {
		using Z = atl::mod_int<uint128, ~uint128(0)>;
		return (Z(~uint128(0) - 1) * Z(~uint128(0) - 1)).value() == 1;
	}());

	static_assert(atl::barrett().mod() == 1 && atl::barrett64().mod() == 1);
	static_assert(atl::barrett(1000000007u).mul(123456, 789012) == uint32(uint64(123456) * 789012 % 1000000007));
	static_assert(atl::barrett64(atl::largest_prime_64).mul(12345678901234ull, 98765432109876ull)
				  == uint64(uint128(12345678901234ull) * 98765432109876ull % atl::largest_prime_64));
	static_assert(atl::barrett(~uint32(0)).mul(~uint32(0) - 1, ~uint32(0) - 1) == 1);
	static_assert(atl::barrett64(~uint64(0)).mul(~uint64(0) - 1, ~uint64(0) - 1) == 1);
	static_assert(atl::barrett(1u).mul(0, 0) == 0 && atl::barrett64(1).mul(0, 0) == 0);

	static_assert([] {
		using mg = atl::montgomery<uint32(998244353)>;
		uint32 const a = 123456789, b = 987654321 % 998244353;
		uint32 const fa = mg::to_form(a), fb = mg::to_form(b);
		uint32 const lazy = mg::mul_lazy(fa, fb), fixed = mg::mul_lazy_fixed(fa, fb, mg::companion(fb));
		return mg::from_form(fa) == a && mg::from_form(mg::mul(fa, fb)) == uint32(uint64(a) * b % 998244353)
			&& lazy % 998244353 == mg::mul(fa, fb) && fixed % 998244353 == mg::mul(fa, fb)
			&& mg::from_form(mg::reduce_lazy(uint64(fa) * fb) % mg::mod) == mg::from_form(mg::mul(fa, fb));
	}());
	static_assert([] {
		using mg = atl::montgomery<atl::largest_prime_64>;
		uint64 const fa = mg::to_form(1), fb = mg::to_form(atl::largest_prime_64 - 1);
		return mg::from_form(mg::sub(fa, fb)) == 2 && mg::from_form(mg::sub(fb, fa)) == atl::largest_prime_64 - 2
			&& mg::from_form(mg::mul(fb, fb)) == 1 && mg::mul(fb, mg::inverse(fb)) == mg::one()
			&& mg::from_form(mg::power(mg::to_form(2), 64)) == 59;
	}());
	static_assert([] {
		using mg = atl::montgomery<atl::uint16(65521)>;
		return mg::from_form(mg::mul(mg::to_form(65520), mg::to_form(65520))) == 1;
	}());
	static_assert([] {
		using mg = atl::montgomery<uint64(1)>;
		return mg::one() == 0 && mg::power(0, 0) == 0 && mg::from_form(mg::to_form(0)) == 0;
	}());
	static_assert([] {
		using mg = atl::montgomery<uint64(3)>;
		uint64 const a = mg::to_form(2);
		return mg::from_form(mg::add(a, a)) == 1 && mg::from_form(mg::neg(a)) == 1
			&& mg::from_form(mg::incr(a)) == 0 && mg::from_form(mg::decr(a)) == 1;
	}());
	static_assert([] {
		using mg = atl::montgomery<(uint64(1) << 62) - 1>;
		uint64 const a = 2 * mg::mod - 1, b = mg::to_form(3);
		return mg::mul_lazy(a, a) % mg::mod == mg::mul(a % mg::mod, a % mg::mod)
			&& mg::mul_lazy_fixed(a, b, mg::companion(b)) % mg::mod == mg::mul(a % mg::mod, b);
	}());
	static_assert([] {
		atl::detail::montgomery64 const mg(atl::largest_prime_64);
		uint64 const a = mg.to_form(atl::largest_prime_64 - 1);
		return mg.mod() == atl::largest_prime_64 && mg.from_form(a) == atl::largest_prime_64 - 1
			&& mg.from_form(mg.mul(a, a)) == 1 && mg.from_form(mg.add(a, mg.one())) == 0
			&& mg.from_form(mg.power(mg.to_form(2), 64)) == 59;
	}());

	static_assert([] {
		using sm = atl::static_modulus<uint32(4294967291u)>;
		return sm::add(4294967290u, 4294967289u) == 4294967288u && sm::sub(1u, 4294967290u) == 2u
			&& sm::neg(1u) == 4294967290u && sm::incr(4294967290u) == 0u && sm::decr(0u) == 4294967290u
			&& sm::mul(4294967290u, 4294967290u) == 1u && sm::power(2u, 32) == 5u && sm::mul(sm::inverse(3u), 3u) == 1u
			&& sm::one() == 1u && sm::from_form(sm::to_form(42u)) == 42u;
	}());
	static_assert(atl::static_modulus<(uint64(1) << 61) - 1>::mul(uint64(1) << 60, 4) == 2);
	static_assert(atl::static_modulus<~uint32(0)>::mul(~uint32(0) - 1, ~uint32(0) - 1) == 1);
	static_assert(atl::static_modulus<~uint64(0)>::mul(~uint64(0) - 1, ~uint64(0) - 1) == 1);

	static_assert(atl::reduce_mod(-1, 7u) == 6u);
	static_assert(atl::reduce_mod(atl::int64(-9223372036854775807LL - 1), 10u) == 2u);
	static_assert(atl::reduce_mod(uint64(1) << 40, 1000u) == uint32((uint64(1) << 40) % 1000));
	static_assert(atl::reduce_mod((uint128(1) << 100) + 1, 1000u) == 377u);
	static_assert(atl::reduce_mod(-atl::int128(uint128(1) << 100), 1000u) == 624u);
	static_assert(atl::add_mod_unchecked(4294967290u, 4294967290u, 4294967291u) == 4294967289u);
	static_assert(atl::sub_mod_unchecked(0u, 4294967290u, 4294967291u) == 1u);
	static_assert(atl::mul_mod_unchecked(4294967290u, 4294967290u, 4294967291u) == 1u);
	static_assert(atl::mul_mod_binary(atl::largest_prime_64 - 1, atl::largest_prime_64 - 1, atl::largest_prime_64)
				  == 1);
	static_assert(atl::add_mod(10u, 12u, 7u) == 1u
				  && atl::sub_mod(3u, 12u, 7u) == 5u
				  && atl::mul_mod(10u, 12u, 7u) == 1u);
	static_assert(atl::incr_mod(6u, 7u) == 0u && atl::decr_mod(0u, 7u) == 6u
				  && atl::neg_mod(2u, 7u) == 5u && atl::neg_mod(0u, 7u) == 0u);
	static_assert(atl::inverse_mod(3u, 7u) == 5u);
	static_assert(atl::inverse_mod(uint128(2), ~uint128(0)) == uint128(1) << 127);
	static_assert(atl::detail::mulhi_u128((uint128(1) << 65) - 1, (uint128(1) << 65) - 1) == 3);
	static_assert([] {
		auto const result = atl::extended_gcd(~uint128(0), ~uint128(0) - 1);
		return result.gcd == 1 && result.x == 1 && result.y == -1;
	}());
	static_assert(atl::pow_mod<uint64>(2, 10, 1000000007) == 1024);
	static_assert(atl::pow_mod<uint64>(3, 1000000006, 1000000007) == 1);
	static_assert(atl::power(atl::int64(2), 62u) == atl::int64(1) << 62);
	static_assert(atl::power(3, 0u) == 1);
	static_assert(atl::power(atl::mod_int<uint32, 7>(3), 0u).value() == 1);
	static_assert(atl::power(3, 4u, 1) == 81);

	static_assert(atl::is_prime(atl::largest_prime_64));
	static_assert([] {
		for (uint64 n = atl::largest_prime_64 + 1; n != 0; ++n)
			if (atl::is_prime(n)) return false;
		return true;
	}());
	static_assert(atl::is_prime(atl::largest_prime_32));
	static_assert([] {
		for (uint64 n = uint64(atl::largest_prime_32) + 1; n < (uint64(1) << 32); ++n)
			if (atl::is_prime(n)) return false;
		return true;
	}());
}

namespace x2_checks {
	static_assert(atl::is_prime(97) && !atl::is_prime(91));
	static_assert(atl::factorization(360).size() == 3);
	static_assert(atl::factorization(360)[0] == atl::prime_factor{2, 3});
	static_assert(atl::factorization(1009 * 1013) == std::vector<atl::prime_factor>{{1009, 1}, {1013, 1}});
	static_assert(atl::experimental::factorization_by_trial_division(360) == atl::factorization(360));
	static_assert(atl::sieve_info(10).size() == 10);
	static_assert(atl::prime_range(20).size() == 8);
	static_assert([] {
		atl::sieve s(100);
		auto const f = s.factorization(72);
		auto const pf = s.prime_factors(90);
		return s.size() == 100 && s.is_prime(29) && !s.is_prime(28) && s.divisors(12).size() == 6
			&& f.size() == 2 && f[0] == atl::prime_factor{2, 3} && f[1] == atl::prime_factor{3, 2}
			&& pf.size() == 3 && pf[2] == 5 && s.min_prime_factor(91) == 7 && s.primes().size() == 25;
	}());
	static_assert([] {
		auto const info = atl::sieve_info(13);
		return info[0].prime == 0 && info[1].prime == 1 && info[3].prime == 3 && info[3].prime_power == 3
			&& info[12].prime == 2 && info[12].exponent == 2 && info[12].prime_power == 4;
	}());
	static_assert(atl::sieve_item_info{2, 2, 4} == atl::sieve_item_info{2, 2, 4}
				  && atl::sieve_item_info{2, 2, 4} != atl::sieve_item_info{2, 1, 2});
	static_assert(atl::euler_totient_range(10)[6] == 2);
	static_assert(atl::divisor_count_range(13)[12] == 6);
	static_assert(atl::divisor_sum_range(13)[12] == 28);
	static_assert(atl::divisor_power_sum_range<atl::uint64>(13, 2)[12] == 210);
	constexpr auto prime_power_rule = [](atl::uint64, atl::uint64 e, atl::uint64, std::vector<atl::uint64> const &) {
		return atl::uint64(1) << e;
	};
	static_assert(atl::multiplicative_fn_range<atl::uint64>(13, prime_power_rule)[12] == 8);
	static_assert(atl::solve_linear_congruence(6, 4, 10u) == atl::congruence<unsigned>{4, 5});
	static_assert(atl::chinese_remainder(atl::congruence<unsigned>{2, 6}, atl::congruence<unsigned>{5, 9})
				  == atl::congruence<unsigned>{14, 18});
	static_assert([] {
		std::array const system{atl::congruence<unsigned>{2, 3}, atl::congruence<unsigned>{3, 5},
								atl::congruence<unsigned>{2, 7}};
		return atl::chinese_remainder(system) == atl::congruence<unsigned>{23, 105}
			&& atl::chinese_remainder_mod(system, 10u) == 3;
	}());
	static_assert(atl::multiplicative_order(2, 7) == 3 && atl::multiplicative_order(0, 1) == 1);
	static_assert(atl::multiplicative_order(3, atl::uint64(1) << 40) == atl::uint64(1) << 38);
	static_assert(atl::multiplicative_order(2, (atl::uint64(1) << 40) - 1) == 40);
	static_assert(atl::detail::ceil_sqrt(~atl::uint64(0)) == atl::uint64(1) << 32);
	static_assert(atl::primitive_root(2) == 1 && atl::primitive_root(17) == 3);
	static_assert(atl::floor_sum(5, 3, 2, 1) == 7 && atl::floor_sum(4, 3, -2, -1) == -7);
	static_assert([] {
		atl::int64 const n = (std::numeric_limits<atl::int64>::max)();
		atl::int128 const pairs = atl::int128(n) * (n - 1) / 2;
		return atl::floor_sum(n, 1, 5, (std::numeric_limits<atl::int64>::min)()) == 3 * pairs - 2 * atl::int128(n)
			&& atl::floor_sum(n, 1, -5, n) == -3 * pairs + n
			&& atl::floor_sum(n, n, n - 1, n - 1) == pairs;
	}());
	static_assert(atl::floor_staircase_product(5, 3, 2, 1, 1u, 10u, 0u,
											   [](unsigned l, unsigned r) { return l + r; })
				  == 53);
	constexpr auto minimum_reference = [](unsigned const &l, unsigned const &r) -> unsigned const & {
		return l < r ? l : r;
	};
	static_assert(atl::floor_staircase_product(5, 3, 2, 1, 1u, 10u, 1000u, minimum_reference) == 1);
	static_assert(atl::first_residue_in_range(31, 6, 0, 11, 19) == 2);
	static_assert(!atl::first_residue_in_range(10, 2, 0, 3, 3));
	static_assert(atl::first_residue_in_range(~atl::uint64(0), ~atl::uint64(0) - 1, 0, 1, 1) == ~atl::uint64(0) - 1);
}

TEST_CASE("nt interfaces constant-evaluate", "[constexpr]") {
	STATIC_REQUIRE(atl::mod_int<atl::uint32, 7>(3).inverse().value() == 5);
	STATIC_REQUIRE(atl::extended_gcd(atl::uint32(12), atl::uint32(18)).gcd == 6);
	STATIC_REQUIRE(atl::dynamic_mod_int<>{}.value() == 0);
}

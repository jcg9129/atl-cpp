#include <atl/experimental/nt/factorization_trial.hpp>
#include <atl/experimental/nt/power_series_variants.hpp>
#include <atl/experimental/nt/square_root_mod_variants.hpp>
#include <atl/nt/basic.hpp>
#include <atl/nt/batch_inverse.hpp>
#include <atl/nt/bitwise_convolution.hpp>
#include <atl/nt/constants.hpp>
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
#include <atl/nt/ntt_engine.hpp>
#include <atl/nt/power.hpp>
#include <atl/nt/power_series.hpp>
#include <atl/nt/series_calculus.hpp>
#include <atl/nt/sieve.hpp>
#include <atl/nt/square_root_mod.hpp>
#include <atl/nt/static_modulus.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <cstddef>
#include <limits>
#include <optional>
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

namespace {
	using TransformZ = atl::mod_int<atl::uint32, 998244353>;
	static_assert(atl::detail::primitive_root<TransformZ>() == TransformZ(3));
	static_assert(atl::scalar_ntt::supports<TransformZ>());
	static_assert(!atl::scalar_ntt::supports<atl::mod_int<atl::uint32, 4294967291u>>());

	static_assert([] {
		auto const holds = []<atl::static_mod_int Element>(std::size_t n, std::size_t m, Element) {
			auto const c = atl::convolution(std::vector<Element>(n, Element(1)), std::vector<Element>(m, Element(2)));
			if (c.size() != n + m - 1) return false;
			for (std::size_t k = 0; k < c.size(); ++k) {
				std::size_t const pairs = (std::min)(k, n - 1) - (k >= m ? k - m + 1 : 0) + 1;
				if (c[k] != Element(2 * pairs)) return false;
			}
			return true;
		};
		return holds(60, 70, TransformZ()) && holds(49, 400, TransformZ()) && holds(5, 3, TransformZ())
			&& holds(60, 60, atl::mod_int<atl::uint32, 1000000007>());
	}());
	static_assert([] {
		auto const c = atl::convolution_uint64(std::vector<atl::uint32>(50, 3), std::vector<atl::uint32>(60, 5));
		return c.size() == 109 && c[0] == 15 && c[49] == 750 && c[108] == 15;
	}());
	static_assert([] {
		constexpr atl::uint64 largest = (std::numeric_limits<atl::uint64>::max)();
		auto const c = atl::convolution_uint64(std::vector<atl::uint64>(49, largest / 49), std::vector<unsigned>(49, 1));
		if (c.size() != 97) return false;
		for (std::size_t k = 0; k < c.size(); ++k) {
			std::size_t const pairs = k < 49 ? k + 1 : 97 - k;
			if (atl::uint128(c[k]) != atl::uint128(largest / 49) * pairs) return false;
		}
		return true;
	}());
	static_assert([] {
		std::vector<TransformZ> a{1, 2, 3, 4, 5, 6, 7, 8};
		auto const original = a;
		atl::ntt(a);
		atl::inverse_ntt(a);
		return a == original;
	}());
	static_assert([] {
		atl::cplx const a{1, 2}, b{3, -1};
		atl::cplx const p = a * b, sum = a + b, difference = a - b, conjugate = a.conj();
		return p.re == 5 && p.im == 5 && sum.re == 4 && sum.im == 1 && difference.re == -2
			&& difference.im == 3 && conjugate.im == -2;
	}());
	static_assert([] {
		std::vector<long long> f{1, 2, 3, 4};
		atl::subset_zeta(f);
		bool has_expected_values = f == std::vector<long long>{1, 3, 4, 10};
		atl::subset_mobius(f);
		atl::superset_zeta(f);
		has_expected_values = has_expected_values && f == std::vector<long long>{10, 6, 7, 4};
		atl::superset_mobius(f);
		atl::walsh_hadamard(f);
		atl::inverse_walsh_hadamard(f);
		return has_expected_values && f == std::vector<long long>{1, 2, 3, 4};
	}());
	static_assert([] {
		std::vector<TransformZ> const a{1, 2, 3, 4}, b{5, 6, 7, 8};
		auto const o = atl::or_convolution(a, b);
		auto const n = atl::and_convolution(a, b);
		auto const x = atl::xor_convolution(a, b);
		auto const subset = atl::subset_convolution(a, b);
		return o[3] == TransformZ(184) && n[3] == TransformZ(32) && x[0] == TransformZ(70) && subset[3] == TransformZ(60);
	}());
	static_assert([] {
		atl::uint32 a[]{1, 2, 3, 4}, b[]{5, 6, 7, 8};
		atl::scalar_ntt::convolve<TransformZ>(a, b, 2);
		atl::uint32 c[]{1, 2, 3, 4};
		atl::scalar_ntt::square<TransformZ>(c, 2);
		return a[0] == 66 && a[1] == 68 && a[2] == 66 && a[3] == 60
			&& c[0] == 26 && c[1] == 28 && c[2] == 26 && c[3] == 20;
	}());
}

static_assert([] {
	std::array<atl::uint128, 2> const a{2, 3};
	std::array<unsigned, 1> const b{5};
	return atl::convolution_uint64(a, b) == std::vector<atl::uint64>{10, 15};
}());

#include <atl/nt/bit_matrix.hpp>
#include <atl/nt/gaussian_elimination.hpp>
#include <atl/nt/linear_recurrence.hpp>
#include <atl/nt/matrix.hpp>
#include <span>
#include <utility>

namespace linear_algebra_checks {
	using mint = atl::mod_int32<998244353>;
	static_assert([] {
		atl::matrix<mint> const fib{{1, 1}, {1, 0}};
		auto const p = atl::power(fib, 10u);
		atl::matrix<atl::int64, atl::min_plus_semiring<atl::int64>> w{{0, 4}, {1, 0}};
		auto const two = atl::power(w, 2u);
		atl::matrix<int> m{{1, 2}, {3, 4}};
		std::vector<int> const v{1, 1};
		m.row(0)[0] = 1;
		m.entries()[3] = 4;
		m += atl::matrix<int>(2, 2);
		m *= atl::matrix<int>::identity(2);
		return p(0, 1) == mint(55) && two(0, 1) == 4 && two(1, 0) == 1
			&& (m * std::span<int const>(v)) == std::vector<int>{3, 7} && m.transpose()(0, 1) == 3
			&& (m + m)(1, 1) == 8 && atl::matrix<int>::identity(2) * m == m
			&& m.rows() == 2 && m.columns() == 2 && std::as_const(m).row(1)[0] == 3
			&& std::as_const(m).entries().size() == 4 && atl::matrix<int>().entries().empty();
	}());
	static_assert([] {
		atl::matrix<mint> const a{{2, 1, 1}, {1, 3, 2}, {1, 0, 0}};
		auto const inv = atl::inverse(a);
		auto const s = atl::solve_linear_system(atl::matrix<mint>{{1, 1, 1}, {2, 2, 2}}, {3, 6});
		using z12 = atl::mod_int32<12>;
		double const det = atl::determinant(atl::matrix<double>{{0, 2}, {3, 0}});
		auto const floating_inverse = atl::inverse(atl::matrix<double>{{2}}, 1e-9);
		auto const floating_solution = atl::solve_linear_system(atl::matrix<double>{{2}}, {4.0}, 1e-9);
		return atl::determinant(a) == mint(-1) && inv && *inv * a == atl::matrix<mint>::identity(3)
			&& atl::rank(atl::matrix<mint>{{1, 2}, {2, 4}}) == 1 && s.is_solvable && !s.is_unique()
			&& s.null_space.rows() == 2
			&& atl::determinant_any_modulus(atl::matrix<z12>{{2, 3}, {4, 3}}).value() == 6
			&& det > -6.000000001 && det < -5.999999999
			&& atl::rank(atl::matrix<double>{{1, 2}, {2, 4}}, 1e-9) == 1
			&& floating_inverse && (*floating_inverse)(0, 0) > 0.499999999
			&& (*floating_inverse)(0, 0) < 0.500000001 && floating_solution.is_unique()
			&& floating_solution.particular[0] > 1.999999999 && floating_solution.particular[0] < 2.000000001;
	}());
	static_assert([] {
		atl::bit_matrix a(2, 3);
		a.set(0, 0), a.set(0, 1), a.set(1, 1), a.set(1, 2);
		a.flip(0, 2);
		a.set(0, 2, false);
		a.row_words(1)[0] = 6;
		auto const s = atl::solve_linear_system(a, {true, false});
		return atl::rank(a) == 2 && s.is_solvable && !s.is_unique() && s.null_space.rows() == 1
			&& (s.particular[0] != s.particular[1]) && s.particular[1] == s.particular[2]
			&& a.columns() == 3 && std::as_const(a).row_words(0)[0] == 3 && a.test(1, 2)
			&& atl::bit_matrix() == atl::bit_matrix(0, 0);
	}());
	static_assert([] {
		std::vector<mint> const fib{0, 1, 1, 2, 3, 5, 8, 13};
		auto const c = atl::berlekamp_massey(fib);
		return c == std::vector<mint>{1, 1} && atl::linear_recurrence_term(c, fib, 90u) == mint(2880067194370816120ull)
			&& atl::rational_series_coefficient(std::vector<mint>{1}, std::vector<mint>{1, -2}, 10u) == mint(1024);
	}());
	static_assert([] {
		std::vector<mint> numerator(64), denominator(65);
		denominator[0] = mint(1);
		denominator[64] = mint(-1);
		for (int i = 0; i < 64; ++i) numerator[std::size_t(i)] = mint(i);
		return atl::rational_series_coefficient(numerator, denominator, 1u) == mint(1);
	}());
	static_assert([] {
		using wide = atl::mod_int64<18446744073709551557ull>;
		atl::matrix<wide> const a{{-1, -2}, {-3, -4}};
		auto const square = a * a;
		return square(0, 0) == wide(7) && square(1, 1) == wide(22)
			&& atl::determinant(a) == wide(-2) && atl::determinant_any_modulus(a) == wide(-2);
	}());
	static_assert(atl::max_plus_semiring<int>::multiply(2, 3) == 5);
	static_assert(atl::max_plus_semiring<int>::multiply(atl::max_plus_semiring<int>::zero(), 3) == atl::max_plus_semiring<int>::zero());
	static_assert(atl::max_min_semiring<int>::multiply(2, 3) == 2);
	static_assert(atl::max_min_semiring<int>::add(2, 3) == 3);
	static_assert(atl::max_min_semiring<int>::one() > atl::max_min_semiring<int>::zero());
}

namespace power_series_checks {
	using Coefficient = atl::mod_int32<17>;
	static_assert([] {
		atl::factorial_table<Coefficient> const table(16);
		atl::factorial_table<Coefficient> const empty(0);
		return table.maximum_index() == 16 && table.factorial(4) == Coefficient(24)
			&& table.inverse_factorial(4) * Coefficient(24) == Coefficient(1)
			&& table.inverse_integer(3) * Coefficient(3) == Coefficient(1)
			&& table.binomial(5, 2) == Coefficient(10) && table.binomial(5, 6) == Coefficient(0)
			&& empty.maximum_index() == 0 && empty.factorial(0) == Coefficient(1)
			&& empty.inverse_factorial(0) == Coefficient(1) && empty.binomial(0, 0) == Coefficient(1);
	}());
	static_assert([] {
		atl::factorial_table<Coefficient> grown(0);
		atl::factorial_table<Coefficient> const whole(16);
		grown.extend(3);
		grown.extend(2);
		grown.extend(16);
		for (std::size_t i = 0; i <= 16; ++i)
			if (grown.factorial(i) != whole.factorial(i) || grown.inverse_factorial(i) != whole.inverse_factorial(i)
				|| (i != 0 && grown.inverse_integer(i) != whole.inverse_integer(i))) return false;
		return grown.maximum_index() == 16;
	}());
	static_assert([] {
		std::vector<Coefficient> const values{2, 4, 8};
		return atl::batch_inverse(values) == std::vector<Coefficient>{9, 13, 15}
			&& atl::batch_inverse(std::vector<Coefficient>{}).empty()
			&& atl::series_derivative(values) == std::vector<Coefficient>{4, 16}
			&& atl::series_derivative(std::vector<Coefficient>{}).empty()
			&& atl::series_integral(std::vector<Coefficient>{}) == std::vector<Coefficient>{0}
			&& atl::series_integral(values) == std::vector<Coefficient>{0, 2, 2, 14}
			&& atl::series_derivative(atl::series_integral(values)) == values;
	}());
	static_assert(atl::square_root_mod(0, 2) == 0);
	static_assert(atl::square_root_mod(1, 2) == 1);
	static_assert(atl::square_root_mod(2, 7) == 3);
	static_assert(atl::square_root_mod(4, 17) == 2);
	static_assert(!atl::square_root_mod(3, 17));
	static_assert(atl::square_root_mod(Coefficient(4)) == Coefficient(2));
	static_assert(atl::square_root_mod(4, 18446744073709551557ull) == 2);
	static_assert(atl::experimental::square_root_mod_by_cipolla(0, 2) == 0);
	static_assert(atl::experimental::square_root_mod_by_cipolla(1, 2) == 1);
	static_assert(atl::experimental::square_root_mod_by_cipolla(2, 7) == 3);
	static_assert(atl::experimental::square_root_mod_by_cipolla(4, 17) == 2);
	static_assert(!atl::experimental::square_root_mod_by_cipolla(3, 17));
	static_assert(atl::experimental::square_root_mod_by_cipolla(Coefficient(4)) == Coefficient(2));
	static_assert(atl::experimental::square_root_mod_by_cipolla(4, 18446744073709551557ull) == 2);

	/** @brief Small series calls and every inverse, logarithm and exponential contender constant-evaluate. */
	constexpr bool small_series_checks() {
		using Field = atl::mod_int32<7>;
		using Vector = std::vector<Field>;
		Vector const unit{1, 1}, zero{0, 1}, empty;
		Vector const inverse{1, 6, 1, 6};
		Vector const logarithm{0, 1, 3, 5};
		Vector const exponential{1, 1, 4, 6};
		return atl::maximum_series_length == (std::size_t{1} << 22)
			&& atl::series_inverse(unit, 4) == inverse
			&& atl::series_division(Vector{1}, unit, 4) == inverse
			&& atl::series_logarithm(unit, 4) == logarithm
			&& atl::series_exponential(zero, 4) == exponential
			&& atl::experimental::series_inverse_by_recurrence(unit, 4) == inverse
			&& atl::experimental::series_inverse_by_plain_newton(unit, 4) == inverse
			&& atl::experimental::series_division_by_inverse_product(Vector{1}, unit, 4) == inverse
			&& atl::experimental::series_logarithm_by_inverse_product(unit, 4) == logarithm
			&& atl::experimental::series_exponential_by_recurrence(zero, 4) == exponential
			&& atl::experimental::series_exponential_by_plain_newton(zero, 4) == exponential
			&& atl::series_inverse(empty, 0).empty()
			&& atl::series_logarithm(empty, 0).empty()
			&& atl::series_exponential(unit, 0).empty()
			&& atl::series_division(unit, empty, 0).empty()
			&& atl::series_logarithm(atl::series_exponential(zero, 7), 7) == Vector{0, 1, 0, 0, 0, 0, 0};
	}

	static_assert(small_series_checks());

	// Each check is a separate constant evaluation: Clang's default step limit is per static_assert.
	template<atl::static_prime_mod_int Field>
	constexpr bool inverse_crosses_cutoff() {
		auto const inverse = atl::series_inverse(std::vector<Field>{1, 1}, 65);
		for (std::size_t i = 0; i < inverse.size(); ++i)
			if (inverse[i] != (i % 2 == 0 ? Field(1) : Field(-1))) return false;
		return true;
	}

	template<atl::static_prime_mod_int Field>
	constexpr bool division_crosses_cutoff() {
		std::vector<Field> const unit{1, 1};
		return atl::series_division(std::vector<Field>{1}, unit, 65) == atl::series_inverse_sparse(unit, 65);
	}

	template<atl::static_prime_mod_int Field>
	constexpr bool logarithm_crosses_cutoff() {
		std::vector<Field> const unit{1, 1};
		return atl::series_logarithm(unit, 65) == atl::series_logarithm_sparse(unit, 65);
	}

	template<atl::static_prime_mod_int Field>
	constexpr bool exponential_crosses_cutoff() {
		std::vector<Field> const zero{0, 1};
		return atl::series_exponential(zero, 65) == atl::series_exponential_sparse(zero, 65);
	}

	static_assert(inverse_crosses_cutoff<atl::mod_int32<998244353>>());
	static_assert(inverse_crosses_cutoff<atl::mod_int32<1000000007>>());
	static_assert(division_crosses_cutoff<atl::mod_int32<998244353>>());
	static_assert(division_crosses_cutoff<atl::mod_int32<1000000007>>());
	static_assert(logarithm_crosses_cutoff<atl::mod_int32<998244353>>());
	static_assert(logarithm_crosses_cutoff<atl::mod_int32<1000000007>>());
	static_assert(exponential_crosses_cutoff<atl::mod_int32<998244353>>());
	static_assert(exponential_crosses_cutoff<atl::mod_int32<1000000007>>());

	/** @brief Powers, roots, polynomial division, the sparse entries and their contenders constant-evaluate, also in characteristics 2 and 3. */
	constexpr bool power_root_checks() {
		using Field = atl::mod_int32<7>;
		using Vector = std::vector<Field>;
		Vector const unit{1, 1}, square{0, 0, 4, 4, 1}, expected{0, 2, 1, 0, 0, 0, 0};
		auto const division = atl::polynomial_division(Vector{1, 2, 1, 0}, unit);
		auto const schoolbook = atl::experimental::polynomial_division_by_schoolbook(Vector{1, 2, 1, 0}, unit);
		if (division.quotient != unit || !division.remainder.empty() || schoolbook.quotient != unit || !schoolbook.remainder.empty()) return false;
		if (atl::series_power(unit, 2, 4) != Vector{1, 2, 1, 0}
			|| atl::series_power_sparse(unit, 2, 4) != Vector{1, 2, 1, 0}
			|| atl::experimental::series_power_by_binary_powering(unit, 2, 4) != Vector{1, 2, 1, 0}) return false;
		if (atl::series_square_root(square, 7) != std::optional<Vector>(expected)
			|| atl::series_square_root_sparse(square, 7) != std::optional<Vector>(expected)
			|| atl::experimental::series_square_root_by_plain_newton(square, 7) != std::optional<Vector>(expected)
			|| atl::experimental::series_square_root_by_logarithm(square, 7) != std::optional<Vector>(expected)) return false;
		if (atl::series_inverse_sparse(unit, 4) != atl::series_inverse(unit, 4)
			|| atl::series_division_sparse(Vector{1}, unit, 4) != atl::series_inverse(unit, 4)
			|| atl::series_logarithm_sparse(unit, 4) != atl::series_logarithm(unit, 4)
			|| atl::series_exponential_sparse(Vector{0, 1}, 4) != atl::series_exponential(Vector{0, 1}, 4)) return false;
		if (atl::series_power_sparse(Vector{0, 0, 1, 1}, 3, 10) != Vector{0, 0, 0, 0, 0, 0, 1, 3, 3, 1}
			|| atl::series_square_root_sparse(Vector{0, 0, 0, 0, 1, 2, 1}, 11) != std::optional<Vector>(Vector{0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0})
			|| atl::series_power_sparse(Vector{}, 0, 9) != Vector{1, 0, 0, 0, 0, 0, 0, 0, 0}) return false;
		using Small = atl::mod_int32<3>;
		if (atl::series_power(std::vector<Small>{1, 1}, 3, 5) != std::vector<Small>{1, 0, 0, 1, 0}) return false;
		using Bit = atl::mod_int32<2>;
		return atl::series_square_root(std::vector<Bit>{1, 0, 1}, 3) == std::optional{std::vector<Bit>{1, 1, 0}}
			&& atl::series_square_root_sparse(std::vector<Bit>{1, 0, 1}, 3) == std::optional{std::vector<Bit>{1, 1, 0}};
	}

	static_assert(power_root_checks());

	/** @brief The Newton-driver contenders and the coupled square root cross their seed precision in constant evaluation. */
	constexpr bool square_root_crosses_cutoff() {
		using Field = atl::mod_int32<998244353>;
		using Vector = std::vector<Field>;
		auto const root = atl::series_square_root(Vector{1, 2, 1}, 65);
		Vector expected(65);
		expected[0] = expected[1] = Field(1);
		return root == std::optional<Vector>(expected);
	}

	constexpr bool inverse_driver_crosses_cutoff() {
		using Field = atl::mod_int32<998244353>;
		std::vector<Field> const unit{1, 1};
		return atl::experimental::series_inverse_by_newton_driver(unit, 65) == atl::series_inverse_sparse(unit, 65);
	}

	constexpr bool exponential_driver_crosses_cutoff() {
		using Field = atl::mod_int32<998244353>;
		std::vector<Field> const zero{0, 1};
		return atl::experimental::series_exponential_by_newton_driver(zero, 65) == atl::series_exponential_sparse(zero, 65);
	}

	static_assert(square_root_crosses_cutoff());
	static_assert(inverse_driver_crosses_cutoff());
	static_assert(exponential_driver_crosses_cutoff());
}

#include <atl/experimental/nt/polynomial_points_variants.hpp>
#include <atl/experimental/nt/series_composition_variants.hpp>
#include <atl/nt/polynomial_points.hpp>
#include <atl/nt/series_composition.hpp>

namespace {
	/** @brief Every point operation and contender preserves its algebra and its lengths in constant evaluation. */
	constexpr bool polynomial_points_checks() {
		using Field = atl::mod_int32<7>;
		using Vector = std::vector<Field>;
		namespace ex = atl::experimental;
		Vector const polynomial{1, 2, 3}, points{0, 1, 2}, values{1, 6, 3};
		Field const shift(1);
		atl::polynomial_point_tree<Field> const tree(points);
		if (tree.point_count() != 3 || tree.evaluate(polynomial) != values || tree.interpolate(values) != polynomial) return false;
		if (!std::ranges::equal(tree.points(), points)) return false;
		if (atl::polynomial_evaluation(polynomial, points) != values
			|| ex::polynomial_evaluation_by_remainder_tree(polynomial, points) != values
			|| ex::polynomial_evaluation_by_horner(polynomial, points) != values
			|| ex::polynomial_evaluation_by_leaf_cutoff(polynomial, points, 1) != values) return false;
		if (atl::polynomial_interpolation(points, values) != polynomial
			|| ex::polynomial_interpolation_by_remainder_tree(points, values) != polynomial
			|| ex::polynomial_interpolation_by_newton_differences(points, values) != polynomial) return false;
		if (atl::polynomial_taylor_shift(polynomial, shift) != Vector{6, 1, 3}
			|| ex::polynomial_taylor_shift_by_divide_and_conquer(polynomial, shift) != Vector{6, 1, 3}
			|| ex::polynomial_taylor_shift_by_horner(polynomial, shift) != Vector{6, 1, 3}) return false;
		if (atl::polynomial_from_roots(points) != Vector{0, 2, 4, 1}
			|| ex::polynomial_from_roots_by_sequential_product(points) != Vector{0, 2, 4, 1}) return false;
		std::vector<Vector> const factors{{1, 1, 0}, {1, 2}};
		if (atl::polynomial_product(factors) != Vector{1, 3, 2}
			|| ex::polynomial_product_by_smallest_first(factors) != Vector{1, 3, 2}
			|| ex::polynomial_product_by_sequential_schoolbook(factors) != Vector{1, 3, 2}) return false;
		if (atl::polynomial_geometric_evaluation(polynomial, Field(1), Field(2), 3) != Vector{6, 3, 1}
			|| ex::polynomial_geometric_evaluation_by_horner(polynomial, Field(1), Field(2), 3) != Vector{6, 3, 1}) return false;
		Vector const expected{1, 6, 3, 6, 1, 2, 2, 1, 6, 3};
		if (atl::sample_point_shift(values, Field(0), 10) != expected
			|| ex::sample_point_shift_by_falling_factorials(values, Field(0), 10) != expected
			|| ex::sample_point_shift_by_direct_barycentric(values, Field(0), 10) != expected) return false;
		Vector beyond_characteristic(9);
		beyond_characteristic[7] = Field(1);
		return atl::polynomial_taylor_shift(beyond_characteristic, Field(1)) == Vector{1, 0, 0, 0, 0, 0, 0, 1, 0};
	}

	static_assert(polynomial_points_checks());

	/** @brief The transform-backed tree directions, which only run above the schoolbook leaf size. */
	constexpr bool polynomial_points_tree_checks() {
		using Field = atl::mod_int32<998244353>;
		using Vector = std::vector<Field>;
		Vector points(33), polynomial(33);
		for (std::size_t i = 0; i < points.size(); ++i) points[i] = Field(i);
		polynomial[0] = Field(1);
		polynomial[32] = Field(2);
		atl::polynomial_point_tree<Field> const tree(points);
		return tree.interpolate(tree.evaluate(polynomial)) == polynomial;
	}

	static_assert(polynomial_points_tree_checks());

	/** @brief Composition, power projection and compositional inverse with every contender, including a precision beyond the characteristic. */
	constexpr bool series_composition_checks() {
		using Field = atl::mod_int32<7>;
		using Vector = std::vector<Field>;
		namespace ex = atl::experimental;
		Vector const outer{1, 2, 3}, inner{1, 1, 1}, weights{1, 2, 3};
		Vector const composed{6, 1, 4, 6};
		if (atl::series_composition(outer, inner, 4) != composed
			|| ex::series_composition_by_brent_kung(outer, inner, 4) != composed
			|| ex::series_composition_by_fast_horner(outer, inner, 4) != composed
			|| ex::series_composition_by_schoolbook_horner(outer, inner, 4) != composed
			|| ex::series_composition_by_cutoffs(outer, inner, 4, 1, 1) != composed) return false;
		Vector const moments{1, 6, 0, 4, 4};
		if (atl::series_power_projection(weights, inner, 5) != moments
			|| ex::series_power_projection_by_padded_bostan_mori(weights, inner, 5) != moments
			|| ex::series_power_projection_by_repeated_products(weights, inner, 5) != moments
			|| ex::series_power_projection_by_bostan_mori(weights, inner, 5) != moments) return false;
		Vector const invertible{0, 1, 1}, inverse{0, 1, 6, 2, 2};
		if (atl::series_compositional_inverse(invertible, 5) != inverse
			|| ex::series_compositional_inverse_by_composition_newton(invertible, 5) != inverse
			|| ex::series_compositional_inverse_by_projection_lagrange(invertible, 5) != inverse
			|| ex::series_compositional_inverse_by_baby_giant_lagrange(invertible, 5) != inverse
			|| ex::series_compositional_inverse_by_coefficient_recurrence(invertible, 5) != inverse) return false;
		auto const beyond = atl::series_compositional_inverse(invertible, 8);
		if (atl::series_composition(invertible, beyond, 8) != Vector{0, 1, 0, 0, 0, 0, 0, 0}) return false;
		using Bit = atl::mod_int32<2>;
		return atl::series_compositional_inverse(std::vector<Bit>{0, 1, 1}, 5) == std::vector<Bit>{0, 1, 1, 0, 1};
	}

	// MSVC does not constant-evaluate the transposed composition, a recursion returning vectors: its
	// checks also run at run time (TEST_CASE "series composition checks at run time").
#if !defined(_MSC_VER) || defined(__clang__)
	static_assert(series_composition_checks());
#endif

	/** @brief The transposed composition (forced by the cutoffs 1) and the half-transform doubling of its spectra. */
	constexpr bool series_composition_transform_checks() {
		using Field = atl::mod_int32<998244353>;
		using Vector = std::vector<Field>;
		Vector const outer{1, 2, 3};
		Vector inner(20);
		inner[1] = Field(1);
		inner[2] = Field(1);
		Vector expected(20);
		expected[0] = Field(1);
		expected[1] = Field(2);
		expected[2] = Field(5);
		expected[3] = Field(6);
		expected[4] = Field(3);
		Vector packed{1, 2, 3};
		auto const transformed = atl::detail::composition_transform<atl::default_ntt_engine, Field>(packed, 8);
		packed.resize(8);
		atl::ntt(packed);
		return atl::series_composition(outer, inner, 20) == expected
			&& atl::experimental::series_composition_by_cutoffs(outer, inner, 20, 1, 1) == expected && packed == transformed;
	}

#if !defined(_MSC_VER) || defined(__clang__)
	static_assert(series_composition_transform_checks());
#endif
}

#include <atl/experimental/nt/online_convolution_variants.hpp>
#include <atl/experimental/nt/polynomial_gcd_variants.hpp>
#include <atl/nt/online_convolution.hpp>
#include <atl/nt/polynomial_gcd.hpp>

namespace {
	/** @brief Gcd, minimal cofactors, modular inverses, Pade approximants and the reversed-sequence recurrence, with every contender. */
	constexpr bool polynomial_gcd_checks() {
		using Field = atl::mod_int32<7>;
		using Vector = std::vector<Field>;
		namespace ex = atl::experimental;
		Vector const first{2, 3, 1}, second{3, 4, 1};
		if (atl::polynomial_gcd(first, second) != Vector{1, 1}) return false;
		auto const extended = atl::polynomial_extended_gcd(first, second);
		if (extended.gcd != Vector{1, 1} || extended.first_cofactor != Vector{6} || extended.second_cofactor != Vector{1}) return false;
		if (!(extended == atl::polynomial_extended_gcd_result<Field>{{1, 1}, {6}, {1}}) || extended == atl::polynomial_extended_gcd_result<Field>{{1, 1}, {6}, {}}) return false;
		if (atl::polynomial_gcd(first, Vector{}) != first || !atl::polynomial_gcd(Vector{}, Vector{0}).empty()) return false;
		if (atl::polynomial_inverse_mod(Vector{0, 1}, Vector{1, 0, 1}) != std::optional<Vector>(Vector{0, 6})) return false;
		if (atl::polynomial_inverse_mod(Vector{}, Vector{1, 1})) return false;
		auto const pade = atl::pade_approximant(Vector{1, 1, 1, 1}, 1, 2);
		if (!pade || pade->numerator != Vector{1} || pade->denominator != Vector{1, 6}) return false;
		if (atl::pade_approximant(Vector{0, 1}, 0, 1)) return false;
		if (ex::polynomial_gcd_by_fused_matrix_products(first, second) != Vector{1, 1}) return false;
		if (ex::polynomial_gcd_by_fast_division_euclid(first, second) != Vector{1, 1}) return false;
		if (ex::polynomial_gcd_by_schoolbook_euclid(first, second) != Vector{1, 1}) return false;
		if (ex::polynomial_gcd_by_leaf_cutoff_plain_products(first, second, 1) != Vector{1, 1}) return false;
		if (ex::polynomial_gcd_by_leaf_cutoff_fused_products(first, second, 1) != Vector{1, 1}) return false;
		if (ex::polynomial_extended_gcd_by_fused_matrix_products(first, second).gcd != Vector{1, 1}) return false;
		if (ex::polynomial_extended_gcd_by_fast_division_euclid(first, second).gcd != Vector{1, 1}) return false;
		if (ex::polynomial_extended_gcd_by_schoolbook_euclid(first, second).gcd != Vector{1, 1}) return false;
		if (ex::polynomial_extended_gcd_by_leaf_cutoff_plain_products(first, second, 1).second_cofactor != Vector{1}) return false;
		if (ex::polynomial_extended_gcd_by_leaf_cutoff_fused_products(first, second, 1).second_cofactor != Vector{1}) return false;
		if (ex::pade_approximant_by_quadratic_euclid(Vector{1, 1, 1}, 0, 2)->denominator != Vector{1, 6}) return false;
		return ex::berlekamp_massey_by_half_gcd(Vector{0, 1, 1, 2, 3, 5}) == Vector{1, 1};
	}

	static_assert(polynomial_gcd_checks());

	/** @brief Every online schedule streams its inputs, with feedback, in constant evaluation. */
	constexpr bool online_convolution_checks() {
		using Field = atl::mod_int32<7>;
		using Vector = std::vector<Field>;
		namespace ex = atl::experimental;
		Vector const kernel{1, 2, 3};
		atl::online_convolution<Field> full;
		atl::online_fixed_convolution<Field> fixed(kernel);
		full.reserve(12);
		fixed.reserve(12);
		ex::online_convolution_by_block_cutoff<Field> block_cutoff(1);
		ex::online_convolution_by_uncached_blocks<Field> uncached(1);
		ex::online_convolution_by_direct_sums<Field> direct;
		ex::online_convolution_by_karatsuba_blocks<Field> karatsuba(1);
		ex::online_fixed_convolution_by_block_cutoff<Field> fixed_block_cutoff(kernel, 1);
		ex::online_fixed_convolution_by_uncached_blocks<Field> fixed_uncached(kernel, 1);
		ex::online_fixed_convolution_by_direct_sums<Field> fixed_direct(kernel);
		ex::online_fixed_convolution_by_karatsuba_blocks<Field> fixed_karatsuba(kernel, 1);
		for (std::size_t i = 0; i < 12; ++i) {
			Field const left(i + 1), right(i < kernel.size() ? kernel[i] : Field(0));
			Field expected(0);
			for (std::size_t j = 0; j <= i && j < kernel.size(); ++j) expected += Field(i - j + 1) * kernel[j];
			if (full.push(left, right) != expected || fixed.push(left) != expected) return false;
			if (block_cutoff.push(left, right) != expected || uncached.push(left, right) != expected || direct.push(left, right) != expected
				|| karatsuba.push(left, right) != expected) return false;
			if (fixed_block_cutoff.push(left) != expected || fixed_uncached.push(left) != expected || fixed_direct.push(left) != expected
				|| fixed_karatsuba.push(left) != expected) return false;
		}
		return full.size() == 12 && fixed.size() == 12 && block_cutoff.size() == 12 && uncached.size() == 12 && direct.size() == 12
			&& karatsuba.size() == 12 && fixed_block_cutoff.size() == 12;
	}

	static_assert(online_convolution_checks());

	/** @brief With a transform-friendly prime and cutoff 1, the block products go through the transforms, cached and not. */
	constexpr bool online_convolution_transform_checks() {
		using Field = atl::mod_int32<998244353>;
		using Vector = std::vector<Field>;
		namespace ex = atl::experimental;
		Vector kernel(4);
		for (std::size_t i = 0; i < kernel.size(); ++i) kernel[i] = Field(2 * i + 1);
		ex::online_convolution_by_block_cutoff<Field> cached(1);
		ex::online_convolution_by_uncached_blocks<Field> uncached(1);
		ex::online_fixed_convolution_by_block_cutoff<Field> fixed(kernel, 1);
		for (std::size_t i = 0; i < 9; ++i) {
			Field const left(i + 3), right(i < kernel.size() ? kernel[i] : Field(0));
			Field expected(0);
			for (std::size_t j = 0; j <= i && j < kernel.size(); ++j) expected += Field(i - j + 3) * kernel[j];
			if (cached.push(left, right) != expected || uncached.push(left, right) != expected || fixed.push(left) != expected) return false;
		}
		return true;
	}

	static_assert(online_convolution_transform_checks());
}

TEST_CASE("series composition checks at run time", "[constexpr]") {
	CHECK(series_composition_checks());
	CHECK(series_composition_transform_checks());
}

#include <atl/config.hpp>
#include <atl/nt/dynamic_mod_int.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <cstdint>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include "../expect_abort.hpp"

using atl::uint32;
using atl::uint64;

TEST_CASE("dynamic_mod_int arithmetic matches a 64-bit reference", "[dynamic_mod_int]") {
	using Z = atl::dynamic_mod_int<>;
	for (uint32 m : {2u, 3u, 998244353u, 1000000007u, 4294967291u}) {
		Z::set_mod(m);
		std::mt19937 rng(m);
		bool ok = true;
		for (int i = 0; i < 50000 && ok; ++i) {
			uint64 a = rng() % m, b = rng() % m;
			Z za = Z(a), zb = Z(b);
			ok &= (za + zb).value() == (a + b) % m;
			ok &= (za - zb).value() == (a + m - b) % m;
			ok &= (za * zb).value() == a * b % m;
		}
		REQUIRE(ok);
	}
}

TEST_CASE("dynamic_mod_int64 arithmetic matches a 128-bit reference", "[dynamic_mod_int]") {
	using Z = atl::dynamic_mod_int64<>;
	using u128 = atl::uint128;
	for (uint64 m : {2ull, 998244353ull, 3000000000000000019ull, 18446744073709551557ull}) {
		Z::set_mod(m);
		std::mt19937_64 rng(m);
		bool ok = true;
		for (int i = 0; i < 50000 && ok; ++i) {
			uint64 a = rng() % m, b = rng() % m;
			Z za = Z(a), zb = Z(b);
			ok &= za.value() == a && zb.value() == b;
			ok &= (za + zb).value() == uint64((u128(a) + b) % m);
			ok &= (za - zb).value() == uint64((u128(a) + m - b) % m);
			ok &= (za * zb).value() == uint64(u128(a) * b % m);
		}
		REQUIRE(ok);
	}
}

TEST_CASE("dynamic_mod_int64 inverse (prime modulus near 2^64)", "[dynamic_mod_int]") {
	using Z = atl::dynamic_mod_int64<>;
	Z::set_mod(18446744073709551557ull);
	std::mt19937_64 rng(1);
	bool ok = true;
	for (int i = 0; i < 5000 && ok; ++i) {
		uint64 a = rng() % 18446744073709551557ull;
		if (a == 0) continue;
		ok &= (Z(a) * Z(a).inverse()).value() == 1;
	}
	REQUIRE(ok);
}

TEST_CASE("dynamic_mod_int64 operator>> reads values past INT64_MAX", "[dynamic_mod_int]") {
	using Z = atl::dynamic_mod_int64<>;
	Z::set_mod(18446744073709551557ull);

	std::istringstream in("10000000000000000000 18446744073709551557");
	Z a, b;
	in >> a >> b;
	REQUIRE(a.value() == 10000000000000000000ull);
	REQUIRE(b.value() == 0);
}

TEST_CASE("dynamic_mod_int64 operator>> reads a negative as mod - v", "[dynamic_mod_int]") {
	using Z = atl::dynamic_mod_int64<1>;
	Z::set_mod(1000000000000000009ull);
	std::istringstream in("-1 -5");
	Z a, b;
	in >> a >> b;
	REQUIRE(a.value() == Z::mod() - 1);
	REQUIRE(b.value() == Z::mod() - 5);
}

TEST_CASE("dynamic_mod_int construction reduces and handles negatives", "[dynamic_mod_int]") {
	using Z = atl::dynamic_mod_int<>;
	Z::set_mod(1000000007u);
	REQUIRE(Z(1000000007LL).value() == 0);
	REQUIRE(Z(1000000008LL).value() == 1);
	REQUIRE(Z(-1).value() == 1000000006u);
	REQUIRE(Z(-1000000008LL).value() == 1000000006u);
}

TEST_CASE("dynamic_mod_int inverse and division (prime modulus)", "[dynamic_mod_int]") {
	using Z = atl::dynamic_mod_int<>;
	Z::set_mod(998244353u);
	std::mt19937 rng(1);
	bool ok = true;
	for (int i = 0; i < 20000 && ok; ++i) {
		uint32 a = rng() % 998244353u;
		if (a == 0) continue;
		Z za = Z(a), zb = Z(rng() % 998244353u);
		ok &= (za * za.inverse()).value() == 1;
		ok &= ((zb / za) * za).value() == zb.value();
	}
	REQUIRE(ok);
}

TEST_CASE("dynamic_mod_int power", "[dynamic_mod_int]") {
	using Z = atl::dynamic_mod_int<>;
	Z::set_mod(1000000007u);

	for (uint32 a : {2u, 3u, 123456u, 999999u})
		REQUIRE(Z(a).power(1000000006ull).value() == 1);
	REQUIRE(Z(2).power(10).value() == 1024);
}

TEST_CASE("dynamic_mod_int increment / unary / compare", "[dynamic_mod_int]") {
	using Z = atl::dynamic_mod_int<>;
	Z::set_mod(5u);
	Z a = 4;
	REQUIRE((++a).value() == 0);
	REQUIRE((a++).value() == 0);
	REQUIRE(a.value() == 1);
	REQUIRE((-Z(1)).value() == 4);
	REQUIRE(Z(3) == Z(8));
	REQUIRE(Z(3) != Z(4));
}

TEST_CASE("dynamic_mod_int operator>> reads and reduces", "[dynamic_mod_int]") {
	using Z = atl::dynamic_mod_int<>;
	Z::set_mod(1000000007u);
	std::istringstream in("5 1000000008 -1");
	Z a, b, c;
	in >> a >> b >> c;
	REQUIRE(a.value() == 5);
	REQUIRE(b.value() == 1);
	REQUIRE(c.value() == 1000000006u);
}

TEST_CASE("dynamic_mod_int distinct Id => independent moduli", "[dynamic_mod_int]") {
	using Z0 = atl::dynamic_mod_int32<0>;
	using Z1 = atl::dynamic_mod_int32<1>;
	Z0::set_mod(1000000007u);
	Z1::set_mod(998244353u);
	REQUIRE(Z0::mod() == 1000000007u);
	REQUIRE(Z1::mod() == 998244353u);
	REQUIRE(Z0(1000000008LL).value() == 1);
	REQUIRE(Z1(998244354LL).value() == 1);
}

TEST_CASE("dynamic_mod_int: set_mod(0) and out-of-range unchecked are asserted", "[dynamic_mod_int]") {
	using Z = atl::dynamic_mod_int32<7>;
	CHECK(atl_test::aborts([] { Z::set_mod(0u); }));
	Z::set_mod(1u);
	REQUIRE((Z(5) * Z(3)).value() == 0);
	Z::set_mod(11u);
	REQUIRE(Z::unchecked(10u).value() == 10u);
	CHECK(atl_test::aborts([] { (void)Z::unchecked(11u); }));

	REQUIRE(Z::unchecked(10).value() == 10u);
	REQUIRE(Z::unchecked(uint64(3)).value() == 3u);
	CHECK(atl_test::aborts([] { (void)Z::unchecked(-1); }));
	CHECK(atl_test::aborts([] { (void)Z::unchecked(uint64(1) << 32); }));
}

TEST_CASE("dynamic_mod_int: 64-bit unsigned inputs reduce exactly under every modulus", "[dynamic_mod_int]") {
	using Z = atl::dynamic_mod_int<atl::uint32, 7>;
	std::mt19937_64 rng(31);
	for (atl::uint32 m : {1u, 2u, 3u, 998244353u, 1000000007u, 2147483648u, 4294967291u, 4294967295u}) {
		Z::set_mod(m);
		for (atl::uint64 z : {atl::uint64(0), atl::uint64(m) - 1, atl::uint64(m), ~atl::uint64(0),
							  ~atl::uint64(0) - 1, atl::uint64(m) * m, atl::uint64(1) << 63})
			REQUIRE(Z(z).value() == z % m);
		for (int i = 0; i < 20000; ++i) {
			atl::uint64 const z = rng() >> (rng() % 64);
			REQUIRE(Z(z).value() == z % m);
			REQUIRE(Z((unsigned long long)z).value() == z % m);
		}
	}
}

TEST_CASE("dynamic_mod_int: 64-bit signed inputs reduce exactly under every modulus", "[dynamic_mod_int]") {
	using Z = atl::dynamic_mod_int<atl::uint32, 8>;
	std::mt19937_64 rng(37);
	for (atl::uint32 m : {1u, 2u, 3u, 998244353u, 2147483648u, 4294967291u, 4294967295u}) {
		Z::set_mod(m);
		auto reference = [&](atl::int64 v) {
			atl::int128 r = atl::int128(v) % m;
			return atl::uint32(r < 0 ? r + m : r);
		};
		for (atl::int64 v : {atl::int64(0), atl::int64(-1), atl::int64(m), -atl::int64(m), INT64_MIN, INT64_MAX,
							 INT64_MIN + 1})
			REQUIRE(Z(v).value() == reference(v));
		for (int i = 0; i < 20000; ++i) {
			atl::int64 const v = atl::int64(rng()) >> (rng() % 63);
			REQUIRE(Z(v).value() == reference(v));
			REQUIRE(Z((long long)v).value() == reference(v));
		}
	}
}

TEST_CASE("dynamic_mod_int covers modulus boundaries and wide source values", "[dynamic_mod_int][boundary]") {
	using D32 = atl::dynamic_mod_int32<92>;
	using D64 = atl::dynamic_mod_int64<92>;
	for (uint32 modulus : {1u, 2147483647u, 4294967291u, ~uint32(0)}) {
		D32::set_mod(modulus);
		REQUIRE((D32(modulus - 1) * D32(modulus - 1)).value() == 1u % modulus);
		REQUIRE(D32((atl::uint128(1) << 100) + 1).value() == atl::reduce_mod((atl::uint128(1) << 100) + 1, modulus));
	}
	for (uint64 modulus : {uint64(1), (uint64(1) << 61) - 1, uint64(18446744073709551557ull), ~uint64(0)}) {
		D64::set_mod(modulus);
		D64 a(modulus - 1), b(modulus - 1);
		REQUIRE((a * b).value() == 1u % modulus);
		REQUIRE((a + b).value() == atl::add_mod_unchecked(modulus - 1, modulus - 1, modulus));
		REQUIRE((D64(0) - b).value() == 1u % modulus);
		REQUIRE((++a).value() == 0);
		REQUIRE((--a).value() == modulus - 1);
	}
	CHECK(atl_test::aborts([] { D64::set_mod(0); }));
}

TEST_CASE("dynamic_mod_int extraction shares a checked contract at both widths", "[dynamic_mod_int][streams]") {
	auto check = []<typename Mint>() {
		Mint::set_mod(1000000007);
		for (auto text : {"-+5", "--5", "- 5", "+ 5", "", "abc", "+", "-", "18446744073709551616", "-18446744073709551616",
						  "340282366920938463463374607431768211455", "-340282366920938463463374607431768211455",
						  "340282366920938463463374607431768211456", "-340282366920938463463374607431768211456"}) {
			Mint value(3);
			std::istringstream input(text);
			input >> value;
			REQUIRE(input.fail());
			REQUIRE(value == Mint(3));
		}
		for (auto text : {"+5", "-5", "5", " 5"}) {
			Mint value(3);
			std::istringstream input(text);
			input >> value;
			REQUIRE(input.rdstate() == std::ios_base::eofbit);
			REQUIRE(value == Mint(text[0] == '-' ? -5 : 5));
		}
		for (auto text : {"18446744073709551615", "-18446744073709551615", "4294967296"}) {
			Mint value(3);
			std::istringstream input(text);
			input >> value;
			REQUIRE(input.rdstate() == std::ios_base::eofbit);
			Mint const expected(text[0] == '4' ? uint64(1) << 32 : (std::numeric_limits<uint64>::max)());
			REQUIRE(value == (text[0] == '-' ? -expected : expected));
		}
		{
			Mint value(3);
			std::istringstream input(" 5");
			input >> std::noskipws >> value;
			REQUIRE(input.fail());
			REQUIRE(value == Mint(3));
		}
		for (auto text : {"0xFf", "ff", "+0XFF", "-0xff"}) {
			Mint value(3);
			std::istringstream input(text);
			input >> std::hex >> value;
			REQUIRE(input.rdstate() == std::ios_base::eofbit);
			REQUIRE(value == (text[0] == '-' ? -Mint(255) : Mint(255)));
		}
		{
			Mint value(3);
			std::istringstream input("17 010");
			input >> std::oct >> value;
			REQUIRE(value == Mint(15));
			input.unsetf(std::ios_base::basefield);
			input >> value;
			REQUIRE(value == Mint(10));
		}
		{
			Mint value(3);
			std::istringstream input("184467440737095516160123!");
			input >> value;
			REQUIRE(input.fail());
			REQUIRE(value == Mint(3));
			input.clear();
			REQUIRE(input.peek() == '!');
		}
		{
			Mint value(3);
			std::istringstream input("1,234");
			input >> value;
			REQUIRE_FALSE(input.fail());
			REQUIRE(value == Mint(1));
			REQUIRE(input.peek() == ',');
		}
		{
			Mint value(3);
			std::istringstream input("5");
			input.exceptions(std::ios_base::eofbit);
			REQUIRE_THROWS_AS(input >> value, std::ios_base::failure);
			REQUIRE(value == Mint(5));
		}
		for (auto text : {"-+5", "18446744073709551616"}) {
			Mint value(3);
			std::istringstream input(text);
			input.exceptions(std::ios_base::failbit);
			auto const flags = input.flags();
			REQUIRE_THROWS_AS(input >> value, std::ios_base::failure);
			REQUIRE(input.flags() == flags);
			REQUIRE(value == Mint(3));
		}
	};
	check.template operator()<atl::dynamic_mod_int32<95>>();
	check.template operator()<atl::dynamic_mod_int64<95>>();
}

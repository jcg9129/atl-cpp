#include <atl/config.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/nt/modular.hpp>
#include <atl/types/alias.hpp>
#include <catch2/catch.hpp>
#include <climits>
#include <concepts>
#include <iomanip>
#include <limits>
#include <locale>
#include <random>
#include <sstream>
#include <streambuf>
#include <string>
#include <type_traits>
#include <utility>
#include "../expect_abort.hpp"

using atl::int64;
using atl::uint32;
using atl::uint64;

namespace {
	constexpr uint32 test_modulus = 1'000'000'007;
	using Z = atl::mod_int<uint32, test_modulus>;
}

TEST_CASE("mod_int construction and reduction", "[modint]") {
	REQUIRE(Z(0).value() == 0);
	REQUIRE(Z(test_modulus).value() == 0);
	REQUIRE(Z(test_modulus + 5).value() == 5);
	REQUIRE(Z::mod() == test_modulus);

	REQUIRE(Z(-1).value() == test_modulus - 1);
	REQUIRE(Z(int64(-1) - int64(test_modulus)).value() == test_modulus - 1);
	REQUIRE((Z(3) == Z(3)));
	REQUIRE((Z(3) != Z(4)));
}

TEST_CASE("mod_int arithmetic matches reference modular arithmetic", "[modint]") {
	auto seed = GENERATE(take(1, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	std::uniform_int_distribution<uint32> dist(0, test_modulus - 1);

	for (int t = 0; t < 5000; ++t) {
		uint32 a = dist(rng), b = dist(rng);
		Z za(a), zb(b);
		int64 m = test_modulus;

		REQUIRE((za + zb).value() == (a + b) % test_modulus);
		REQUIRE((za - zb).value() == ((int64(a) - int64(b)) % m + m) % m);
		REQUIRE((za * zb).value() == uint64(a) * b % test_modulus);
		REQUIRE((-za).value() == (m - int64(a)) % m);
		REQUIRE((+za) == za);

		Z acc = za;
		acc += zb;
		REQUIRE(acc == za + zb);
		acc -= zb;
		REQUIRE(acc == za);

		if (b != 0) {
			Z q = za / zb;
			REQUIRE(q * zb == za);
			REQUIRE((zb * zb.inverse()) == Z(1));
		}
	}
}

TEST_CASE("mod_int increment and decrement", "[modint]") {
	Z z(test_modulus - 1);
	REQUIRE((++z).value() == 0);
	REQUIRE((z++).value() == 0);
	REQUIRE(z.value() == 1);
	REQUIRE((--z).value() == 0);
	REQUIRE((z--).value() == 0);
	REQUIRE(z.value() == test_modulus - 1);
}

TEST_CASE("mod_int operator>> reads and reduces", "[modint]") {
	std::istringstream in("5 1000000008 -1");
	Z a, b, c;
	in >> a >> b >> c;
	REQUIRE(a.value() == 5);
	REQUIRE(b.value() == 1);
	REQUIRE(c.value() == test_modulus - 1);
}

TEST_CASE("mod_int operator>> reads a negative as modulus minus magnitude for a 64-bit modulus", "[modint]") {
	using Z64 = atl::mod_int<uint64, 1000000000000000009ull>;
	std::istringstream in("-1 -1000000000000000010 -0 18446744073709551615");
	Z64 a, b, c, d;
	in >> a >> b >> c >> d;
	REQUIRE(a.value() == Z64::mod() - 1);
	REQUIRE(b.value() == Z64::mod() - 1);
	REQUIRE(c.value() == 0);
	REQUIRE(d == Z64((std::numeric_limits<uint64>::max)()));
}

TEST_CASE("mod_int over a 64-bit Montgomery modulus", "[modint]") {
	using atl::uint128;
	constexpr uint64 M = 1'000'000'000'000'000'003ull;
	using Z64 = atl::mod_int<uint64, M>;
	auto seed = GENERATE(take(1, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	for (int t = 0; t < 20000; ++t) {
		uint64 a = rng() % M, b = rng() % M;
		Z64 za(a), zb(b);
		REQUIRE(za.value() == a);
		REQUIRE((za + zb).value() == (a + b) % M);
		REQUIRE((za - zb).value() == (a + M - b) % M);
		REQUIRE((za * zb).value() == uint64((uint128(a) * b) % M));
		REQUIRE((-za).value() == (M - a) % M);
		if (b != 0) {
			REQUIRE((za / zb) * zb == za);
			REQUIRE((zb * zb.inverse()) == Z64(1));
		}
	}
	Z64 z(M - 1);
	REQUIRE((++z).value() == 0);
	REQUIRE((--z).value() == M - 1);
}

TEST_CASE("mod_int over a 128-bit modulus multiplies correctly", "[modint]") {
	using atl::uint128;
	constexpr uint128 M = (uint128(1) << 100) + 277;
	using Z128 = atl::mod_int<uint128, M>;
	std::mt19937_64 rng(2024);
	for (int t = 0; t < 5000; ++t) {
		uint128 a = ((uint128(rng()) << 64) | rng()) % M;
		uint128 b = ((uint128(rng()) << 64) | rng()) % M;
		REQUIRE(((Z128(a) * Z128(b)).value() == atl::mul_mod_binary<uint128>(a, b, M)));
	}
}

TEST_CASE("mod_int power matches repeated multiplication", "[modint]") {
	using mint = atl::mod_int32<1'000'000'007>;
	static_assert(std::is_same_v<decltype(++std::declval<mint &>()), mint &>);
	static_assert(std::is_same_v<decltype(--std::declval<mint &>()), mint &>);
	mint base = 12345;
	mint by_mul = 1;
	for (int e = 0; e <= 40; ++e) {
		REQUIRE(base.power(atl::uint64(e)) == by_mul);
		by_mul *= base;
	}
	REQUIRE(mint(0).power(0) == mint(1));

	using mint64 = atl::mod_int64<(atl::uint64(1) << 61) - 87>;
	mint64 b64 = 987654321;
	mint64 acc = 1;
	for (int e = 0; e <= 20; ++e) {
		REQUIRE(b64.power(atl::uint64(e)) == acc);
		acc *= b64;
	}
}

TEST_CASE("mod_int: modulus above half the type range", "[mod_int]") {
	// Wrapping the addition would shift the answer by exactly 2^32 modulo this modulus.
	using ZBig = atl::mod_int<atl::uint32, 4294967291u>;
	ZBig const a(4294967290u), b(4294967289u);
	REQUIRE((a + b).value() == 4294967288u);
	REQUIRE((a + ZBig(0u)).value() == 4294967290u);
	REQUIRE((a - b).value() == 1u);
	REQUIRE((b - a).value() == 4294967290u);
	REQUIRE((ZBig(1u) - a).value() == 2u);
	REQUIRE((a * b).value() == 2u);
	ZBig acc(0u);
	for (int i = 0; i < 1000; ++i) acc += a;
	REQUIRE(acc.value() == 4294967291u - 1000u);
}

TEST_CASE("mod_int: the all-ones modulus of the full type width", "[mod_int]") {
	// A full-width Mersenne fold would lose the extra bit of lo + hi and turn this square into zero.
	using Z32 = atl::mod_int<atl::uint32, 4294967295u>;
	using Z64 = atl::mod_int<atl::uint64, 18446744073709551615ull>;
	static_assert((Z32(4294967294u) * Z32(4294967294u)).value() == 1u);
	static_assert((Z64(18446744073709551614ull) * Z64(18446744073709551614ull)).value() == 1u);
	REQUIRE((Z32(4294967294u) * Z32(4294967294u)).value() == 1u);
	REQUIRE((Z64(18446744073709551614ull) * Z64(18446744073709551614ull)).value() == 1u);

	auto seed = GENERATE(take(8, random(uint64(0), (std::numeric_limits<uint64>::max)())));
	std::mt19937_64 rng(seed);
	for (int round = 0; round < 2000; ++round) {
		uint32 const a = uint32(rng() % 4294967295u), b = uint32(rng() % 4294967295u);
		REQUIRE((Z32(a) * Z32(b)).value() == uint32(uint64(a) * b % 4294967295u));
		uint64 const c = rng() % 18446744073709551615ull, d = rng() % 18446744073709551615ull;
		REQUIRE((Z64(c) * Z64(d)).value() == uint64(atl::uint128(c) * d % 18446744073709551615ull));
	}

	using M31 = atl::mod_int<atl::uint32, 2147483647u>;
	static_assert((M31(2147483646u) * M31(2147483646u)).value() == 1u);
}

TEST_CASE("mod_int: unchecked asserts its residue range", "[mod_int]") {
	REQUIRE(Z::unchecked(test_modulus - 1).value() == test_modulus - 1);
	REQUIRE(Z::unchecked(0).value() == 0);
	CHECK(atl_test::aborts([] { (void)Z::unchecked(test_modulus); }));
	CHECK(atl_test::aborts([] { (void)Z::unchecked(-1); }));
}

namespace {
	struct grouping_facet : std::numpunct<char> {
		std::string pattern;

		explicit grouping_facet(std::string groups = "\3") : pattern(std::move(groups)) {}

		std::string do_grouping() const override { return pattern; }

		char do_thousands_sep() const override { return ','; }
	};

	struct short_output_buffer : std::streambuf {
		std::streamsize xsputn(char const *, std::streamsize) override { return 0; }
	};
}

TEST_CASE("mod_int uint128 insertion honors integer formatting", "[modint][streams]") {
	using atl::uint128;
	using Wide = atl::mod_int<uint128, (uint128(1) << 89) - 1>;
	using Full = atl::mod_int<uint128, (std::numeric_limits<uint128>::max)()>;
	std::ostringstream output;
	output << Wide(0) << ' ' << Wide(1) << ' ' << Wide(uint128(1) << 64);
	REQUIRE(output.str() == "0 1 18446744073709551616");
	output.str("");
	output << Full((std::numeric_limits<uint128>::max)()) << ' ' << Full((std::numeric_limits<uint128>::max)() - 1);
	REQUIRE(output.str() == "0 340282366920938463463374607431768211454");
	output.str("");
	output << std::showbase << std::hex << Wide(0) << ' ' << Wide(1) << ' ' << Wide(uint128(1) << 64)
		   << ' ' << std::uppercase << Wide(0xabcdef);
	REQUIRE(output.str() == "0 0x1 0x10000000000000000 0XABCDEF");
	output.str("");
	output << std::oct << Wide(0) << ' ' << Wide(1) << ' ' << Wide(uint128(1) << 64);
	REQUIRE(output.str() == "0 01 02000000000000000000000");

	for (auto adjustment : {std::ios_base::left, std::ios_base::right, std::ios_base::internal}) {
		output.str("");
		output << std::hex << std::nouppercase;
		output.setf(adjustment, std::ios_base::adjustfield);
		output.fill('_');
		output.width(10);
		output << Wide(0x123);
		REQUIRE(output.width() == 0);
		output << Wide(1);
		REQUIRE(output.str() == (adjustment == std::ios_base::left ? "0x123_____0x1" : adjustment == std::ios_base::right ? "_____0x1230x1"
																														  : "0x_____1230x1"));
	}

	output.str("");
	output.imbue(std::locale(std::locale::classic(), new grouping_facet));
	output << std::dec << Wide(uint128(1) << 64);
	REQUIRE(output.str() == "18,446,744,073,709,551,616");
	output.str("");
	output << std::hex << std::uppercase << Wide(0xabcdef);
	REQUIRE(output.str() == "0XABC,DEF");
	output.str("");
	output << std::oct << Wide(01234567);
	REQUIRE(output.str() == "01,234,567");

	for (auto const &pattern : {std::string("\1"), std::string("\3\2"), std::string("\3\0\2", 3), std::string("\3") + char(CHAR_MAX)}) {
		for (auto base : {std::ios_base::dec, std::ios_base::hex, std::ios_base::oct}) {
			std::ostringstream actual, expected;
			std::locale const locale(std::locale::classic(), new grouping_facet(pattern));
			for (auto *stream : {&actual, &expected}) {
				stream->imbue(locale);
				stream->setf(base, std::ios_base::basefield);
				// Standard libraries disagree on whether the octal prefix joins the first group.
				if (base != std::ios_base::oct) *stream << std::showbase;
				*stream << std::uppercase;
			}
			actual << Wide((std::numeric_limits<uint64>::max)());
			expected << (std::numeric_limits<uint64>::max)();
			REQUIRE(actual.str() == expected.str());
			actual.str("");
			actual << Full((std::numeric_limits<uint128>::max)() - 1);
			REQUIRE_FALSE(actual.fail());
		}
	}
	output.str("");
	output.imbue(std::locale::classic());
	output.width(8);
	output << std::internal << std::oct << Wide(0123);
	REQUIRE(output.str() == "0____123");

	short_output_buffer buffer;
	std::ostream failing(&buffer);
	failing << Wide(1);
	REQUIRE(failing.bad());
	failing.clear();
	failing.exceptions(std::ios_base::badbit);
	REQUIRE_THROWS_AS(failing << Wide(1), std::ios_base::failure);
}

TEST_CASE("mod_int extraction shares a checked contract at every width", "[modint][streams]") {
	auto check = []<atl::unsigned_integral Word, Word Mod>() {
		using Mint = atl::mod_int<Word, Mod>;
		using Accumulator = std::conditional_t<std::same_as<Word, atl::uint128>, atl::uint128, uint64>;
		for (auto text : {"-+5", "--5", "- 5", "+ 5", "", "abc", "+", "-", "0x"}) {
			Mint value(3);
			std::istringstream input(text);
			if (std::string(text) == "0x") input >> std::hex;
			input >> value;
			REQUIRE(input.fail());
			REQUIRE(value == Mint(3));
		}
		for (auto text : {"+5", "-5", "0", "-0", " 5"}) {
			Mint value(3);
			std::istringstream input(text);
			input >> value;
			REQUIRE(input.rdstate() == std::ios_base::eofbit);
			REQUIRE(value == Mint(std::string(text) == "0" || std::string(text) == "-0" ? 0 : text[0] == '-' ? -5
																											 : 5));
		}
		for (bool is_negative : {false, true}) {
			std::string const maximum = sizeof(Word) == 16 ? "340282366920938463463374607431768211455" : "18446744073709551615";
			std::istringstream input((is_negative ? "-" : "") + maximum);
			Mint value(3);
			input >> value;
			REQUIRE(input.rdstate() == std::ios_base::eofbit);
			Mint const expected((std::numeric_limits<Accumulator>::max)());
			REQUIRE(value == (is_negative ? -expected : expected));
			for (auto magnitude : {"18446744073709551616", "340282366920938463463374607431768211456"}) {
				std::istringstream oversized(std::string(is_negative ? "-" : "") + magnitude + "123!");
				value = Mint(3);
				oversized >> value;
				bool const should_fail = sizeof(Word) != 16 || magnitude[0] == '3';
				REQUIRE(oversized.fail() == should_fail);
				if (should_fail) REQUIRE(value == Mint(3));
				oversized.clear();
				REQUIRE(oversized.peek() == '!');
			}
		}
		for (bool is_negative : {false, true}) {
			Mint value(3);
			std::istringstream input(std::string(is_negative ? "-" : "") + "18446744073709551616");
			input >> value;
			if constexpr (std::same_as<Word, atl::uint128>) {
				REQUIRE(input.rdstate() == std::ios_base::eofbit);
				Mint const expected(atl::uint128(1) << 64);
				REQUIRE(value == (is_negative ? -expected : expected));
			} else {
				REQUIRE(input.fail());
				REQUIRE(value == Mint(3));
			}
		}
		for (bool is_negative : {false, true}) {
			for (auto magnitude : {"340282366920938463463374607431768211455", "340282366920938463463374607431768211456"}) {
				Mint value(3);
				std::istringstream input(std::string(is_negative ? "-" : "") + magnitude);
				input >> value;
				bool const should_fail = sizeof(Word) != 16 || std::string(magnitude).back() == '6';
				REQUIRE(input.fail() == should_fail);
				REQUIRE(input.eof());
				if (should_fail)
					REQUIRE(value == Mint(3));
				else {
					Mint const expected((std::numeric_limits<atl::uint128>::max)());
					REQUIRE(value == (is_negative ? -expected : expected));
				}
			}
		}
		{
			Mint value(3);
			std::istringstream input("4294967296");
			input >> value;
			REQUIRE_FALSE(input.fail());
			REQUIRE(value == Mint(uint64(1) << 32));
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
			std::istringstream input("1,234");
			input.imbue(std::locale(std::locale::classic(), new grouping_facet));
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
		for (auto text : {"-+5", "340282366920938463463374607431768211456"}) {
			Mint value(3);
			std::istringstream input(text);
			input.exceptions(std::ios_base::failbit);
			auto const flags = input.flags();
			REQUIRE_THROWS_AS(input >> value, std::ios_base::failure);
			REQUIRE(input.flags() == flags);
			REQUIRE(value == Mint(3));
		}
	};
	check.template operator()<atl::uint8, 251>();
	check.template operator()<atl::uint16, 65521>();
	check.template operator()<uint32, 1'000'000'007>();
	check.template operator()<uint64, 1'000'000'000'000'000'003ull>();
	check.template operator()<atl::uint128, (atl::uint128(1) << 89) - 1>();
}

#include <atl/io/fast_io.hpp>
#include <atl/nt/dynamic_mod_int.hpp>
#include <atl/nt/mod_int.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <atl/types/traits/integral.hpp>
#include <algorithm>
#include <catch2/catch.hpp>
#include <charconv>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <system_error>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include <vector>

using atl::int64;
using atl::uint64;

namespace {
	// Reference decimal formatter, independent of the writer.
	template<atl::integral T>
	std::string ref_str(T v) {
		using U = atl::traits::make_unsigned_t<T>;
		bool neg = false;
		U u;
		if constexpr (atl::signed_integral<T>) {
			neg = v < 0;
			u = neg ? U(U(0) - U(v)) : U(v);
		} else {
			u = v;
		}
		std::string s;
		do {
			s += char('0' + unsigned(u % 10));
			u /= 10;
		} while (u);
		if (neg) s += '-';
		std::ranges::reverse(s);
		return s;
	}

	template<atl::integral T>
	void roundtrip(std::vector<T> const &vals) {
		std::FILE *fp = std::tmpfile();
		REQUIRE(fp != nullptr);
		{
			atl::io::writer w(fp);
			for (T v : vals) {
				w.put(v);
				w.put('\n');
			}
		}
		std::rewind(fp);
		{
			atl::io::reader r(fp);
			for (T v : vals) REQUIRE(r.read_one<T>() == v);
		}
		std::rewind(fp);
		{
			atl::io::reader r(fp);
			for (T v : vals) REQUIRE(r.read_one<std::string>() == ref_str(v));
		}
		std::fclose(fp);
	}

	template<atl::integral T, std::uniform_random_bit_generator Rng>
	std::vector<T> random_fill(std::vector<T> seed, Rng &rng, int extra) {
		for (int i = 0; i < extra; ++i) seed.push_back(T(rng() >> (rng() % 64)));
		return seed;
	}
}

TEST_CASE("io: integer writer/reader round trip", "[io]") {
	std::mt19937_64 rng(0xA71);

	roundtrip(random_fill<uint64>({0, 1, 9, 10, 99, 100, UINT64_MAX, UINT64_MAX - 1}, rng, 20000));
	roundtrip(random_fill<int64>({0, 1, -1, INT64_MIN, INT64_MAX, -123456789}, rng, 20000));
	roundtrip(random_fill<atl::uint8>({0, 1, 255}, rng, 1000));
	roundtrip(random_fill<atl::int8>({0, 1, -1, -128, 127}, rng, 1000));
	roundtrip(random_fill<unsigned>({0u, 1u, UINT_MAX}, rng, 20000));
	roundtrip(random_fill<int>({0, 1, -1, INT_MIN, INT_MAX}, rng, 20000));

	using atl::int128;
	using atl::uint128;
	uint128 const e19 = 10'000'000'000'000'000'000ull;
	std::vector<uint128> u128 = {0, 1, uint128(UINT64_MAX) + 1, uint128(-1), e19 - 1, e19, e19 + 1, e19 * e19 - 1,
								 e19 * e19, e19 * e19 + 7, e19 * 10 + 5};
	for (int i = 0; i < 10000; ++i) u128.push_back(((uint128(rng()) << 64) | rng()) >> (rng() % 128));
	roundtrip(u128);
	std::vector<int128> i128 = {0, 1, -1, std::numeric_limits<int128>::min(), std::numeric_limits<int128>::max(),
								-int128(uint128(1) << 100)};
	for (int i = 0; i < 10000; ++i) i128.push_back(int128((uint128(rng()) << 64) | rng()) >> (rng() % 127));
	roundtrip(i128);
}

TEST_CASE("io: strings, chars, mixed types, negatives", "[io]") {
	std::FILE *fp = std::tmpfile();
	REQUIRE(fp != nullptr);
	std::fputs("  10\t20\r\n hello   world \nx\n-7 -8\n", fp);
	std::rewind(fp);
	atl::io::reader r(fp);

	REQUIRE(r.read_one<int>() == 10);
	REQUIRE(r.read_one<int>() == 20);
	REQUIRE(r.read_one<std::string>() == "hello");
	REQUIRE(r.read_one<std::string>() == "world");
	REQUIRE(r.read_one<char>() == 'x');
	int a = 0, b = 0;
	r >> a >> b;
	REQUIRE(a == -7);
	REQUIRE(b == -8);
	std::fclose(fp);
}

TEST_CASE("io: bool round trip", "[io]") {
	std::FILE *fp = std::tmpfile();
	REQUIRE(fp != nullptr);
	std::vector<bool> vals = {true, false, true, true, false, false, true};
	{
		atl::io::writer w(fp);
		for (bool v : vals) {
			w.put(v);
			w.put(' ');
		}
	}
	std::rewind(fp);
	atl::io::reader r(fp);
	for (bool v : vals) REQUIRE(r.read_one<bool>() == v);
	std::fclose(fp);

	std::FILE *g = std::tmpfile();
	REQUIRE(g != nullptr);
	std::fputs("0 1 5", g);
	std::rewind(g);
	atl::io::reader r2(g);
	REQUIRE(r2.read_one<bool>() == false);
	REQUIRE(r2.read_one<bool>() == true);
	REQUIRE(r2.read_one<bool>() == true);
	std::fclose(g);
}

#if defined(__cpp_lib_to_chars)
TEST_CASE("io: floating point", "[io]") {
	std::mt19937_64 rng(0xF10A7);
	std::vector<double> vals = {0.0, 1.0, -1.0, 3.14159265358979, -2.5, 1e-9, 1e9, 0.1, 123456.789};
	for (int i = 0; i < 5000; ++i) {
		std::uint64_t bits = rng();
		double d;
		std::memcpy(&d, &bits, sizeof(d));
		if (std::isfinite(d)) vals.push_back(d);
	}

	std::FILE *fp = std::tmpfile();
	REQUIRE(fp != nullptr);
	{
		atl::io::writer w(fp);
		for (double v : vals) {
			w.put(v);
			w.put('\n');
		}
	}
	std::rewind(fp);
	{
		atl::io::reader r(fp);
		for (double v : vals) REQUIRE(r.read_one<double>() == v);
	}
	std::fclose(fp);

	std::FILE *g = std::tmpfile();
	REQUIRE(g != nullptr);
	std::vector<double> fx = {3.14159265358979, -2.5, 0.0, 123.456, 1.0 / 3.0, -0.0001, 2.5, 0.125};
	{
		atl::io::writer w(g);
		for (double v : fx) {
			w.put_fixed(v, 6);
			w.put('\n');
		}
	}
	std::rewind(g);
	{
		atl::io::reader r(g);
		for (double v : fx) {
			char buf[64];
			std::snprintf(buf, sizeof(buf), "%.6f", v);
			std::string got = r.read_one<std::string>();
			double a = 0, b = 0;
			std::from_chars(got.data(), got.data() + got.size(), a);
			std::from_chars(buf, buf + std::strlen(buf), b);
			REQUIRE(a == b);
		}
	}
	std::fclose(g);

	std::FILE *h = std::tmpfile();
	REQUIRE(h != nullptr);
	std::vector<float> fs = {0.0f, 1.5f, -3.25f, 1e-6f, 12345.678f};
	{
		atl::io::writer w(h);
		for (float v : fs) {
			w.put(v);
			w.put('\n');
		}
	}
	std::rewind(h);
	{
		atl::io::reader r(h);
		for (float v : fs) REQUIRE(r.read_one<float>() == v);
	}
	std::fclose(h);
}

TEST_CASE("io: long double at full precision", "[io]") {
	// Binary128 needs up to ~45 characters in shortest form, exceeding a
	// 32-byte scratch buffer.
	std::vector<long double> vals = {1.0L / 3.0L, -2.0L / 7.0L, std::numeric_limits<long double>::min(),
									 -std::numeric_limits<long double>::denorm_min(),
									 std::numeric_limits<long double>::max(), 0.1L, -1e-300L};
	std::FILE *fp = std::tmpfile();
	REQUIRE(fp != nullptr);
	{
		atl::io::writer w(fp);
		for (long double v : vals) w.writeln(v);
	}
	std::rewind(fp);
	{
		atl::io::reader r(fp);
		for (long double v : vals) REQUIRE(r.read_one<long double>() == v);
	}
	std::fclose(fp);
}

#endif

TEST_CASE("io: EOF detection", "[io]") {
	std::FILE *fp = std::tmpfile();
	REQUIRE(fp != nullptr);
	std::fputs("3 1 4 1 5   ", fp); // trailing whitespace, no newline
	std::rewind(fp);
	atl::io::reader r(fp);

	int x = 0, count = 0;
	long long sum = 0;
	while (r.try_read(x)) {
		++count;
		sum += x;
	}
	REQUIRE(count == 5);
	REQUIRE(sum == 14);
	REQUIRE(r.eof());

	int untouched = 999;
	REQUIRE_FALSE(r.try_read(untouched)); // past EOF
	REQUIRE(untouched == 999);
	std::fclose(fp);
}

TEST_CASE("io: an integer stops at the first non-digit, on every parsing path", "[io]") {
	// Checking only the high nibble admits ':' through '?' (0x3A-0x3F) as
	// digits 10-15; every parsing width must stop before them.
	std::FILE *fp = std::tmpfile();
	REQUIRE(fp != nullptr);
	std::fputs("12:34:56 7:05 123456789;1 1234<5 99=98>97?96 12345678:12345678\n", fp);
	std::rewind(fp);
	atl::io::reader r(fp);
	for (long long const want : {12, 34, 56}) {
		REQUIRE(r.read<long long>() == want);
		if (want != 56) REQUIRE(r.read<char>() == ':');
	}
	REQUIRE(r.read<int>() == 7);
	REQUIRE(r.read<char>() == ':');
	REQUIRE(r.read<int>() == 5);
	REQUIRE(r.read<unsigned long long>() == 123456789ull);
	REQUIRE(r.read<char>() == ';');
	REQUIRE(r.read<int>() == 1);
	REQUIRE(r.read<int>() == 1234);
	REQUIRE(r.read<char>() == '<');
	REQUIRE(r.read<int>() == 5);
	for (int const want : {99, 98, 97, 96}) {
		REQUIRE(r.read<int>() == want);
		if (want != 96) (void)r.read<char>();
	}
	REQUIRE(r.read<long long>() == 12345678);
	REQUIRE(r.read<char>() == ':');
	REQUIRE(r.read<long long>() == 12345678);
	std::fclose(fp);
}

TEST_CASE("io: a token that is not a number is consumed, so read loops terminate", "[io]") {
	// A zero result must still consume a malformed token so EOF loops advance.
	std::FILE *fp = std::tmpfile();
	REQUIRE(fp != nullptr);
	std::fputs("12 abc 7 -x 9 - 4\n", fp);
	std::rewind(fp);
	atl::io::reader r(fp);
	std::vector<int> seen;
	int x = 0;
	while (r.try_read(x) && seen.size() < 100) seen.push_back(x);
	REQUIRE(seen == std::vector<int>{12, 0, 7, 0, 9, 0, 4});
	REQUIRE(r.eof());
	std::fclose(fp);

	fp = std::tmpfile();
	REQUIRE(fp != nullptr);
	std::fputs("5 -3 +2 8\n", fp); // '-' and '+' are not digits to an unsigned read
	std::rewind(fp);
	{
		atl::io::reader u(fp);
		std::vector<unsigned> values;
		unsigned y = 0;
		while (u.try_read(y) && values.size() < 100) values.push_back(y);
		REQUIRE(values == std::vector<unsigned>{5, 0, 0, 8});
	}
	std::fclose(fp);
}

TEST_CASE("io: flush delivers the bytes, not just to stdio's buffer", "[io]") {
	std::FILE *sink = std::tmpfile();
	REQUIRE(sink != nullptr);
	// An independent FILE buffer observes the underlying file without a seek or
	// read on sink, either of which would flush sink and hide a missing fflush.
#ifdef _WIN32
	int const descriptor = ::_dup(::_fileno(sink));
	REQUIRE(descriptor >= 0);
	std::FILE *observer = ::_fdopen(descriptor, "rb");
#else
	int const descriptor = ::dup(::fileno(sink));
	REQUIRE(descriptor >= 0);
	std::FILE *observer = ::fdopen(descriptor, "rb");
#endif
	REQUIRE(observer != nullptr);
	char stdio_buffer[1 << 16];
	REQUIRE(std::setvbuf(sink, stdio_buffer, _IOFBF, sizeof stdio_buffer) == 0);
	{
		atl::io::writer w(sink);
		w.writeln(42);
		w.flush();
		std::rewind(observer);
		char received[8] = {};
		REQUIRE(std::fread(received, 1, sizeof received, observer) == 3);
		REQUIRE(std::string(received, 3) == "42\n");
	}
	std::fclose(observer);
	std::fclose(sink);
}

TEST_CASE("io: ranges and strings", "[io]") {
	std::FILE *fp = std::tmpfile();
	REQUIRE(fp != nullptr);
	{
		atl::io::writer w(fp);
		w.writeln(std::vector<int>{1, 2, 3});
		w.writeln(std::vector<int>{});
		w.writeln(std::vector<std::string>{"ab", "c"});
		w.writeln(std::string("verbatim string"));
	}
	std::rewind(fp);
	char text[128]{};
	REQUIRE(std::fread(text, 1, sizeof text - 1, fp) > 0);
	REQUIRE(std::string(text) == "1 2 3\n\nab c\nverbatim string\n");
	std::fclose(fp);
}

#if defined(__cpp_lib_to_chars)
TEST_CASE("io: fixed formatting handles the full long double exponent range", "[io]") {
	std::FILE *fp = std::tmpfile();
	REQUIRE(fp != nullptr);
	long double const value = (std::numeric_limits<long double>::max)();
	std::vector<char> expected(std::size_t(std::numeric_limits<long double>::max_exponent10) + 512);
	auto const converted = std::to_chars(
		expected.data(), expected.data() + expected.size(), value, std::chars_format::fixed, 400);
	REQUIRE(converted.ec == std::errc{});
	{
		atl::io::writer w(fp);
		w.put_fixed(value, 400);
	}
	std::rewind(fp);
	{
		atl::io::reader r(fp);
		REQUIRE(r.read_token() == std::string(expected.data(), converted.ptr));
	}
	std::fclose(fp);
}
#endif

TEST_CASE("io: long strings and malformed tokens cross refill boundaries", "[io]") {
	std::FILE *fp = std::tmpfile();
	REQUIRE(fp != nullptr);
	std::string const long_token((1 << 18) + 9, 'x');
	{
		atl::io::writer w(fp);
		w.writeln(long_token, 7);
	}
	std::rewind(fp);
	{
		atl::io::reader r(fp);
		REQUIRE(r.read_token() == long_token);
		REQUIRE(r.read<int>() == 7);
	}
	std::rewind(fp);
	{
		atl::io::reader r(fp);
		REQUIRE(r.read<int>() == 0);
		REQUIRE(r.read<int>() == 7);
		REQUIRE(r.eof());
	}
	std::fclose(fp);
}

TEST_CASE("io: modular integers and ranges", "[io]") {
	using Z = atl::mod_int<atl::uint32, 998244353>;
	using Z64 = atl::mod_int<atl::uint64, (atl::uint64(1) << 61) + 3>;
	using D = atl::dynamic_mod_int32<5>;
	D::set_mod(7u);
	static_assert(atl::io::readable<Z> && atl::io::readable<Z64> && atl::io::readable<D>);
	static_assert(atl::io::writable<Z> && atl::io::writable<Z64> && atl::io::writable<D>);

	std::FILE *fp = std::tmpfile();
	REQUIRE(fp != nullptr);
	{
		atl::io::writer w(fp);
		w.writeln(Z(5), Z(-1), D(10));
		w.writeln(std::vector<Z>{Z(1), Z(2), Z(3)});
		w.writeln(std::vector<int>{});
		w.writeln(std::vector<std::string>{"ab", "c"});
		w.writeln(std::string("verbatim string"));
		w.put("-1 998244354 10 -1 18446744073709551615\n");
	}
	std::rewind(fp);
	char text[256] = {};
	REQUIRE(std::fread(text, 1, sizeof text - 1, fp) > 0);
	REQUIRE(std::string(text)
			== "5 998244352 3\n1 2 3\n\nab c\nverbatim string\n"
			   "-1 998244354 10 -1 18446744073709551615\n");
	std::rewind(fp);
	{
		atl::io::reader r(fp);
		for (int i = 0; i < 10; ++i) (void)r.read<std::string>();
		REQUIRE(r.read<Z>() == Z(998244352));
		REQUIRE(r.read<Z>() == Z(1));
		REQUIRE(r.read<D>().value() == 3u);
		// Separating sign from magnitude makes -1 the modulus minus one at every width.
		REQUIRE(r.read<Z64>() == Z64(Z64::mod() - 1));
		REQUIRE(r.read<Z64>() == Z64((std::numeric_limits<uint64>::max)()));
		REQUIRE(r.eof());
	}
	std::fclose(fp);
}

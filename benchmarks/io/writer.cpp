/**
 * @file
 * Race between ways of formatting unsigned decimal integers into memory, against
 * the library's `io::writer`.
 *
 * All contenders produce the same digits; they differ in where the digits go:
 * - `put_scratch_copy` writes the digits backward into a scratch buffer and then
 *   copies them to the output; for `uint128` it does this with 128-bit divisions
 *   by 100 ("128-bit digit pairs");
 * - `put_in_place` counts the digits first, then writes them backward straight
 *   into the output, which is the scheme `io::writer` uses for up to 64 bits;
 * - `put_chunks_1e19` splits a `uint128` into 19-digit chunks, divides by 10^19
 *   in 128 bits only, and formats each chunk with `put_in_place`.
 * The library row runs `io::writer` on the same values into the null device, so it
 * also includes the cost of the buffered stdio flush.
 *
 * Each benchmark formats a whole vector, space-separated, into a preallocated
 * buffer, so it measures throughput. Shapes: 1e7 uint32 and 1e7 uint64 of mixed
 * lengths, 1e7 uint32 of nine digits, and 1e6 full-width uint128.
 *
 * Result (Xeon Platinum 8280, GCC 15.2, `-O2` and `-O2 -march=native`, 2026-10-15):
 * `put_in_place` beats `put_scratch_copy` on 32- and 64-bit values, where scratch +
 * copy is 6-37% slower. 128-bit digit pairs are 1.9-2.2x slower than
 * `put_chunks_1e19`, which ties the library. Numbers: docs/fast-io.md.
 */
#include <atl/io/writer.hpp>
#include <atl/types/alias.hpp>
#include <atl/types/concepts.hpp>
#include <array>
#include <bit>
#include <catch2/catch.hpp>
#include <concepts>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <ranges>
#include <string_view>
#include <vector>

using namespace atl;

namespace {
#ifdef ATL_IO_BIG_LUT
	// The comparison still needs the two-digit helpers when the library writer
	// selects the four-digit table and omits its private ones.
	constexpr auto digit_pairs = [] {
		std::array<char, 200> table{};
		for (std::size_t i = 0; i < 100; ++i) {
			table[2 * i] = char('0' + i / 10);
			table[2 * i + 1] = char('0' + i % 10);
		}
		return table;
	}();

	template<atl::unsigned_integral U>
	char *digits_backward(char *end, U value) {
		while (value >= 100) {
			auto const pair = uint32(value % 100) * 2;
			value /= 100;
			*--end = digit_pairs[pair + 1];
			*--end = digit_pairs[pair];
		}
		if (value >= 10) {
			auto const pair = uint32(value) * 2;
			*--end = digit_pairs[pair + 1];
			*--end = digit_pairs[pair];
		} else {
			*--end = char('0' + uint32(value));
		}
		return end;
	}

	uint32 decimal_length(uint64 value) {
		constexpr uint64 powers[] = {1ull, 10ull, 100ull, 1000ull, 10000ull, 100000ull, 1000000ull, 10000000ull, 100000000ull, 1000000000ull,
									 10000000000ull, 100000000000ull, 1000000000000ull, 10000000000000ull, 100000000000000ull, 1000000000000000ull,
									 10000000000000000ull, 100000000000000000ull, 1000000000000000000ull, 10000000000000000000ull};
		value |= 1;
		auto const estimate = (uint32(std::bit_width(value)) * 1233) >> 12;
		return estimate + 1 - (value < powers[estimate]);
	}
#else
	using atl::io::detail::decimal_length;
	using atl::io::detail::digits_backward;
#endif

	/** @brief Contender: digits written backward into a scratch buffer, then copied to @p out. */
	template<atl::unsigned_integral U>
	char *put_scratch_copy(char *out, U x) {
		char tmp[44];
		char *const p = digits_backward(tmp + sizeof(tmp), x);
		std::size_t const len = std::size_t(tmp + sizeof(tmp) - p);
		std::memcpy(out, p, len);
		return out + len;
	}

	/** @brief Contender: digit count found first, digits written backward directly into @p out. */
	char *put_in_place(char *out, uint64 x) {
		uint32 const len = decimal_length(x);
		digits_backward(out + len, x);
		return out + len;
	}

	/** @brief Contender: `uint128` split into 19-digit chunks, so only the splitting divisions are 128-bit. */
	char *put_chunks_1e19(char *out, uint128 x) {
		constexpr uint64 chunk = 10000000000000000000ull; // 10^19
		if (x < chunk) return put_in_place(out, uint64(x));
		uint64 const low = uint64(x % chunk);
		uint128 const rest = x / chunk;
		if (rest < chunk) {
			out = put_in_place(out, uint64(rest));
		} else {
			out = put_in_place(out, uint64(rest / chunk));
			char *const mid_end = out + 19;
			char *p = digits_backward(mid_end, uint64(rest % chunk));
			while (p > out) *--p = '0';
			out = mid_end;
		}
		char *const end = out + 19;
		char *p = digits_backward(end, low);
		while (p > out) *--p = '0';
		return end;
	}
}

TEST_CASE("writer integer formatting", "[writer]") {
	std::mt19937_64 rng(11);
	std::vector<uint32> u32_mixed(10'000'000), u32_fixed(10'000'000);
	std::vector<uint64> u64_mixed(10'000'000);
	for (auto &x : u32_mixed) x = uint32(rng() >> (32 + rng() % 32));
	for (auto &x : u32_fixed) x = 100'000'000 + uint32(rng() % 900'000'000);
	for (auto &x : u64_mixed) x = rng() >> (rng() % 64);
	std::vector<uint128> u128(1'000'000);
	for (auto &x : u128) x = (uint128(rng()) << 64) | rng();
	std::vector<char> buffer(260'000'000);

	for (uint32 x : u32_mixed) {
		char a[24], b[24];
		REQUIRE(std::string_view(a, put_scratch_copy(a, x)) == std::string_view(b, put_in_place(b, x)));
	}
	for (uint32 x : u32_fixed) {
		char a[24], b[24];
		REQUIRE(std::string_view(a, put_scratch_copy(a, x)) == std::string_view(b, put_in_place(b, x)));
	}
	for (uint64 x : u64_mixed) {
		char a[24], b[24];
		REQUIRE(std::string_view(a, put_scratch_copy(a, x)) == std::string_view(b, put_in_place(b, x)));
	}
	for (uint128 x : u128) {
		char a[48], b[48];
		REQUIRE(std::string_view(a, put_scratch_copy(a, x)) == std::string_view(b, put_chunks_1e19(b, x)));
	}

#ifdef _WIN32
	std::FILE *const sink = std::fopen("NUL", "wb");
#else
	std::FILE *const sink = std::fopen("/dev/null", "wb");
#endif
	REQUIRE(sink != nullptr);
	/** @brief Library: `io::writer` to the null device, each value followed by a space; returns the value count. */
	auto library_writer = [&]<std::ranges::sized_range R>(R const &values)
		requires std::ranges::input_range<R const> && std::ranges::sized_range<R const> && io::writable<std::ranges::range_value_t<R const>>
	{
		io::writer w(sink);
		for (auto x : values) {
			w.put(x);
			w.put(' ');
		}
		w.flush();
		return std::ranges::size(values);
	};
	/** @brief Format every value with @p put into the shared buffer, space-separated; returns the byte count. */
	auto format_to_memory = [&]<std::ranges::input_range R, typename Formatter>(R const &values, Formatter put)
		requires std::ranges::input_range<R const> && std::regular_invocable<Formatter &, char *, std::ranges::range_reference_t<R const>>
			  && std::same_as<std::invoke_result_t<Formatter &, char *, std::ranges::range_reference_t<R const>>, char *>
	{
		char *out = buffer.data();
		for (auto x : values) {
			out = put(out, x);
			*out++ = ' ';
		}
		return std::size_t(out - buffer.data());
	};
	BENCHMARK("1e7 uint32 mixed lengths | library (io::writer to null device)") { return library_writer(u32_mixed); };
	BENCHMARK("1e7 uint32 mixed lengths | scratch + copy") {
		return format_to_memory(u32_mixed, [](char *o, uint32 x) { return put_scratch_copy(o, x); });
	};
	BENCHMARK("1e7 uint32 mixed lengths | in place (count digits first)") {
		return format_to_memory(u32_mixed, [](char *o, uint32 x) { return put_in_place(o, x); });
	};
	BENCHMARK("1e7 uint32 9 digits | scratch + copy") {
		return format_to_memory(u32_fixed, [](char *o, uint32 x) { return put_scratch_copy(o, x); });
	};
	BENCHMARK("1e7 uint32 9 digits | in place (count digits first)") {
		return format_to_memory(u32_fixed, [](char *o, uint32 x) { return put_in_place(o, x); });
	};
	BENCHMARK("1e7 uint64 mixed lengths | library (io::writer to null device)") { return library_writer(u64_mixed); };
	BENCHMARK("1e7 uint64 mixed lengths | scratch + copy") {
		return format_to_memory(u64_mixed, [](char *o, uint64 x) { return put_scratch_copy(o, x); });
	};
	BENCHMARK("1e7 uint64 mixed lengths | in place (count digits first)") {
		return format_to_memory(u64_mixed, [](char *o, uint64 x) { return put_in_place(o, x); });
	};
	BENCHMARK("1e6 uint128 full width | library (io::writer to null device)") { return library_writer(u128); };
	BENCHMARK("1e6 uint128 full width | 128-bit digit pairs") {
		return format_to_memory(u128, [](char *o, uint128 x) { return put_scratch_copy(o, x); });
	};
	BENCHMARK("1e6 uint128 full width | 10^19 chunks") {
		return format_to_memory(u128, [](char *o, uint128 x) { return put_chunks_1e19(o, x); });
	};
	std::fclose(sink);
}
